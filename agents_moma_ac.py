"""
agents_moma_ac.py
=================
MOMA-AC: Multi-Objective Actor-Critic voi Preference Conditioning (TD3 backbone)
- PreferencedActor  : obs || omega concatenation → split sigmoid/tanh output
- MultiHeadActor    : shared trunk, one output head per UAV (MOMA-AC Sec. 4.2)
- VectorCritic      : twin critics output R^n_obj
- ReplayBuffer      : luu omega theo tung transition
- PrioritizedReplayBuffer : GPI-based priorities (GPI-PD, Eq. 10 + 17)
- GPILinearSupport  : weight support M, corner weights (GPI-LS, Alg. 1)
- optimistic_value  : LP upper bound on the value at a weight (OLS)
- DynamicsEnsemble  : learned model for Dyna updates (GPI-PD, Eq. 13-14)
- MOMA_AC_Agent     : main class, omega_sampling='random'|'ols'|'gpi_ls'|'gpi_pd'
- MOMA_CTDE_Agent   : several UAVs, multi-headed actor + centralised critics (MOMA-AC Alg. 1)
"""

import os
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from typing import List, Optional

# MOMA_DEVICE overrides the choice, e.g. to train on CPU on a machine with a GPU
DEVICE = torch.device(os.environ.get("MOMA_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))

# Evaluation episodes reuse these seeds (common random numbers), so values stay
# comparable across omegas, checkpoints and sampling variants.
EVAL_SEED = 12345
SLA_KEYS = ("sla_embb", "sla_urllc", "sla_mmtc")   # info: share of UEs meeting their full SLA, per slice


def sla_of(info: dict) -> list:
    return [info.get(k, np.nan) for k in SLA_KEYS]


def nan_mean(rows) -> np.ndarray:
    """Column means that skip nan (a slice with no demand that step); nan where a column has no value."""
    a = np.asarray(rows, dtype=np.float64)
    a = a.reshape(len(a), -1) if a.size else np.full((0, len(SLA_KEYS)), np.nan)
    ok = ~np.isnan(a)
    n = ok.sum(axis=0)
    return np.where(n > 0, np.where(ok, a, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)


# ============================================================
# Actor  (Concatenation conditioning — obs || omega → MLP)
# ============================================================

class PreferencedActor(nn.Module):
    """
    Preference-conditioned actor: concatenate omega with obs as input.
    Split output: tanh for position dims, sigmoid for power and BW dims.
    """
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int = 3,
                 hidden: int = 256, n_pose_dims: int = 3):
        super().__init__()
        self.n_pose_dims = n_pose_dims
        inp = obs_dim + n_obj          # concat omega → simple, proven to work
        self.fc1 = nn.Linear(inp, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([obs, omega], dim=-1)       # [obs || omega]
        h   = F.relu(self.ln1(self.fc1(x)))
        h   = F.relu(self.ln2(self.fc2(h)))
        raw = self.out(h)
        # Split activation per semantic:
        pose = torch.tanh(raw[..., :self.n_pose_dims])    # position:    [-1, 1]
        rest = torch.sigmoid(raw[..., self.n_pose_dims:]) # power + BW:  [ 0, 1]
        return torch.cat([pose, rest], dim=-1)


class MultiHeadActor(nn.Module):
    """
    One network for all UAVs (MOMA-AC Sec. 4.2): a shared trunk maps a UAV's own
    obs || omega to features, and that UAV's head maps the features to its action.
    Same split output as PreferencedActor.
    """
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int, n_obj: int = 3,
                 hidden: int = 256, n_pose_dims: int = 3):
        super().__init__()
        self.n_agents = n_agents
        self.n_pose_dims = n_pose_dims
        inp = obs_dim + n_obj
        self.fc1 = nn.Linear(inp, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.heads = nn.ModuleList(nn.Linear(hidden, act_dim) for _ in range(n_agents))

    def forward(self, obs: torch.Tensor, omega: torch.Tensor, agent_idx: int) -> torch.Tensor:
        x   = torch.cat([obs, omega], dim=-1)       # [obs_i || omega]
        h   = F.relu(self.ln1(self.fc1(x)))
        h   = F.relu(self.ln2(self.fc2(h)))
        raw = self.heads[agent_idx](h)
        pose = torch.tanh(raw[..., :self.n_pose_dims])    # position:    [-1, 1]
        rest = torch.sigmoid(raw[..., self.n_pose_dims:]) # power + BW:  [ 0, 1]
        return torch.cat([pose, rest], dim=-1)


# ============================================================
# Critic  (concatenation conditioning, output R^n_obj)
# ============================================================

class VectorCritic(nn.Module):
    """Q(obs, action, omega) -> R^n_obj (one Q-value per objective).
    omega_dim is the width of the omega input: n_obj, or n_agents * n_obj with one omega per UAV."""
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int = 3, hidden: int = 256, omega_dim: int = None):
        super().__init__()
        inp = obs_dim + act_dim + (n_obj if omega_dim is None else omega_dim)
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_obj)

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                omega: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action, omega], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)   # (B, n_obj)


# ============================================================
# Replay Buffers
# ============================================================

# ============================================================
# Attention networks for any number of UAVs (thesis item B2)
# ============================================================

N_OBS_BASE = 30   # Observation dims before the other-UAV block with N > 1: 5 state + 9 DAs + 4 handover + 8 + 4
N_OBS_OTHER = 4   # Per other UAV: dx, dy, dh, power ratio (0 for a UAV out of the mission)


def split_obs(obs: torch.Tensor, n_agents: int):
    """(..., obs_dim) -> own features (..., self_dim) and other UAVs (..., n_agents - 1, 4), in id order.
    The own features are the dims before the other-UAV block plus those after it (backhaul, wind)."""
    end = N_OBS_BASE + N_OBS_OTHER * (n_agents - 1)
    own = torch.cat([obs[..., :N_OBS_BASE], obs[..., end:]], dim=-1)
    others = obs[..., N_OBS_BASE:end].reshape(*obs.shape[:-1], n_agents - 1, N_OBS_OTHER)
    return own, others


class AttentionActor(nn.Module):
    """
    pi(o_i, omega_i, omega_j) for any number of UAVs, one network shared by all of them.
    Query: an MLP of UAV i's own features and omega_i. Keys and values: one token per other UAV (its dx, dy, dh,
    power ratio and omega_j, as in AA-MAMORL's tokens), masked when the UAV is out of the mission (power 0), plus
    one learned null token so that a UAV alone still attends to something. Output as PreferencedActor.
    """
    def __init__(self, self_dim: int, act_dim: int, n_obj: int = 3, hidden: int = 128, n_heads: int = 4,
                 n_pose_dims: int = 3):
        super().__init__()
        self.n_pose_dims = n_pose_dims
        self.q_net = nn.Sequential(nn.Linear(self_dim + n_obj, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        self.kv_net = nn.Sequential(nn.Linear(N_OBS_OTHER + n_obj, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.LayerNorm(2 * hidden), nn.ReLU())
        self.out = nn.Linear(2 * hidden, act_dim)

    def forward(self, obs: torch.Tensor, omegas: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """obs (B, obs_dim) of UAV agent_idx; omegas (B, N, n_obj), every UAV's omega"""
        B, N = omegas.shape[0], omegas.shape[1]
        own, others = split_obs(obs, N)
        q = self.q_net(torch.cat([own, omegas[:, agent_idx]], dim=-1)).unsqueeze(1)          # (B, 1, H)
        idx = [j for j in range(N) if j != agent_idx]
        kv = self.kv_net(torch.cat([others, omegas[:, idx]], dim=-1)) if idx else q[:, :0]   # (B, N-1, H)
        kv = torch.cat([self.null.expand(B, 1, -1), kv], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs.device),
                          others[..., 3] == 0], dim=1)                                        # True: ignore
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
        raw = self.out(self.trunk(torch.cat([q, ctx], dim=-1).squeeze(1)))
        pose = torch.tanh(raw[..., :self.n_pose_dims])
        rest = torch.sigmoid(raw[..., self.n_pose_dims:])
        return torch.cat([pose, rest], dim=-1)

    def forward_all(self, obs: torch.Tensor, omegas: torch.Tensor) -> torch.Tensor:
        """Every UAV's action in one pass: obs (B, N, obs_dim), omegas (B, N, n_obj) -> (B, N, act_dim).
        The same function as forward for each agent_idx, batched over the UAVs."""
        B, N = omegas.shape[0], omegas.shape[1]
        own, others = split_obs(obs, N)                                       # (B,N,self), (B,N,N-1,4)
        q = self.q_net(torch.cat([own, omegas], dim=-1)).reshape(B * N, 1, -1)
        idx = torch.tensor([[j for j in range(N) if j != i] for i in range(N)], dtype=torch.long,
                           device=obs.device).reshape(N, N - 1)
        kv = self.kv_net(torch.cat([others, omegas[:, idx]], dim=-1)).reshape(B * N, N - 1, self.null.shape[-1])
        kv = torch.cat([self.null.expand(B * N, 1, -1), kv], dim=1)
        mask = torch.cat([torch.zeros(B * N, 1, dtype=torch.bool, device=obs.device),
                          (others[..., 3] == 0).reshape(B * N, N - 1)], dim=1)
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=mask, need_weights=False)
        raw = self.out(self.trunk(torch.cat([q, ctx], dim=-1).squeeze(1))).reshape(B, N, -1)
        pose = torch.tanh(raw[..., :self.n_pose_dims])
        rest = torch.sigmoid(raw[..., self.n_pose_dims:])
        return torch.cat([pose, rest], dim=-1)


class AttentionCritic(nn.Module):
    """
    Q(o, a, omega) -> R^n_obj for any number of UAVs. One token per UAV: its own features (absolute position,
    power, battery, demand areas, ...), its action and its omega. A transformer layer mixes the tokens of the UAVs
    in the mission (a UAV with power 0 is masked), their mean goes to the Q head, so Q is invariant to the order
    of the UAVs and the same weights serve any N. omega is one team omega (B, n_obj) or one per UAV (B, N n_obj).
    One layer by default: on a GTX 1650 an update (batch 256, 3 UAVs) takes 30 ms against 43 ms with two layers
    and 14 ms for the MLP networks.
    """
    def __init__(self, self_dim: int, act_dim: int, n_obj: int = 3, hidden: int = 128, n_heads: int = 4,
                 n_layers: int = 1):
        super().__init__()
        self.self_dim, self.act_dim, self.n_obj = self_dim, act_dim, n_obj
        self.embed = nn.Sequential(nn.Linear(self_dim + act_dim + n_obj, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, n_heads, 2 * hidden, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_obj))

    def forward(self, obs: torch.Tensor, action: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        N = action.shape[-1] // self.act_dim
        o = obs.view(B, N, -1)
        own, _ = split_obs(o, N)
        w = omega.view(B, N, self.n_obj) if omega.shape[-1] == N * self.n_obj else \
            omega.unsqueeze(1).expand(B, N, self.n_obj)
        x = self.embed(torch.cat([own, action.view(B, N, self.act_dim), w], dim=-1))
        inactive = o[..., 3] == 0                                    # Own power ratio 0: out of the mission
        h = self.encoder(x, src_key_padding_mask=inactive)
        keep = (~inactive).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int,
                 max_size: int = 500_000, omega_dim: int = None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.action   = np.zeros((max_size, act_dim),  dtype=np.float32)
        self.r_vec    = np.zeros((max_size, n_obj),    dtype=np.float32)
        self.obs_next = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.omega    = np.zeros((max_size, n_obj if omega_dim is None else omega_dim), dtype=np.float32)
        self.done     = np.zeros((max_size, 1),        dtype=np.float32)

    def add(self, obs, action, r_vec, obs_next, omega, done):
        i = self.ptr
        self.obs[i]      = obs
        self.action[i]   = action
        self.r_vec[i]    = r_vec
        self.obs_next[i] = obs_next
        self.omega[i]    = omega
        self.done[i]     = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, action, r_vec, obs_next, omega):
        """Add non-terminal transitions in one call (model rollouts)."""
        n = len(obs)
        i = (self.ptr + np.arange(n)) % self.max_size
        self.obs[i]      = obs
        self.action[i]   = action
        self.r_vec[i]    = r_vec
        self.obs_next[i] = obs_next
        self.omega[i]    = omega
        self.done[i]     = 0.0
        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        return np.random.randint(0, self.size, batch_size)

    def get(self, idx: np.ndarray):
        def t(arr): return torch.FloatTensor(arr[idx]).to(DEVICE)
        return t(self.obs), t(self.action), t(self.r_vec), \
               t(self.obs_next), t(self.omega), t(self.done)

    def __len__(self):
        return self.size

    def _state(self) -> dict:
        n = self.size
        return {"ptr": self.ptr, "size": n,
                "obs": self.obs[:n], "action": self.action[:n], "r_vec": self.r_vec[:n],
                "obs_next": self.obs_next[:n], "omega": self.omega[:n], "done": self.done[:n]}

    def save(self, path: str):
        np.savez(path, **self._state())

    def load(self, path: str):
        with np.load(path) as st:
            self.ptr, self.size = int(st["ptr"]), int(st["size"])
            for key in ("obs", "action", "r_vec", "obs_next", "omega", "done"):
                getattr(self, key)[:self.size] = st[key]
            self._load_extra(st)

    def _load_extra(self, st):
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Proportional sampling with a sum tree. The stored priority is
    max(|delta|^alpha, kappa) (Eq. 17 of GPI-PD, from Fujimoto et al. 2020).
    """
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int,
                 max_size: int = 500_000, alpha: float = 0.6, kappa: float = 0.1):
        super().__init__(obs_dim, act_dim, n_obj, max_size)
        self.alpha = alpha
        self.kappa = kappa
        self.capacity = 1 << int(np.ceil(np.log2(max_size)))
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.max_priority = 1.0

    def _set_priorities(self, idx: np.ndarray, priorities: np.ndarray):
        pos = idx + self.capacity
        self.tree[pos] = priorities
        pos = np.unique(pos // 2)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos = np.unique(pos // 2)

    def add(self, obs, action, r_vec, obs_next, omega, done):
        i = self.ptr
        super().add(obs, action, r_vec, obs_next, omega, done)
        self._set_priorities(np.array([i]), np.array([self.max_priority]))

    def sample_indices(self, batch_size: int) -> np.ndarray:
        # Stratified: one draw from each of batch_size equal slices of the total
        u = (np.arange(batch_size) + np.random.random(batch_size)) / batch_size * self.tree[1]
        pos = np.ones(batch_size, dtype=np.int64)
        while pos[0] < self.capacity:
            left = 2 * pos
            go_right = u >= self.tree[left]
            u = u - self.tree[left] * go_right
            pos = left + go_right
        return np.minimum(pos - self.capacity, self.size - 1)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray):
        priorities = np.maximum(np.abs(td_errors) ** self.alpha, self.kappa)
        self.max_priority = max(self.max_priority, float(priorities.max()))
        # A batch can hold an index twice; keep one priority per index
        idx, first = np.unique(idx, return_index=True)
        self._set_priorities(idx, priorities[first])

    def _state(self) -> dict:
        return {**super()._state(), "tree": self.tree, "max_priority": self.max_priority}

    def _load_extra(self, st):
        self.tree[:] = st["tree"]
        self.max_priority = float(st["max_priority"])


# ============================================================
# GPI Linear Support
# ============================================================

def corner_weights(values: np.ndarray) -> np.ndarray:
    """
    Corner weights of a value set: the weight parts of the vertices of the
    polyhedron in Eq. 6 of GPI-PD, {(w, u) | V w <= u, sum(w) = 1, w >= 0}.
    Enumerates vertices directly, which is cheap for a small V and 3 objectives.
    """
    V = np.asarray(values, dtype=np.float64)
    n, d = V.shape
    G = np.vstack([
        np.hstack([V, -np.ones((n, 1))]),
        np.hstack([-np.eye(d), np.zeros((d, 1))]),
    ])
    combos = np.array(list(itertools.combinations(range(n + d), d)))
    A = np.zeros((len(combos), d + 1, d + 1))
    A[:, :d, :] = G[combos]
    A[:, d, :d] = 1.0
    b = np.zeros((len(combos), d + 1, 1))
    b[:, d, 0] = 1.0

    regular = np.abs(np.linalg.det(A)) > 1e-12
    x = np.linalg.solve(A[regular], b[regular])[..., 0]
    feasible = np.all(x @ G.T <= 1e-9, axis=1)
    return np.unique(x[feasible, :d].round(4) + 0.0, axis=0)    # + 0.0 turns -0. into 0.


def optimistic_value(w: np.ndarray, weights: np.ndarray, best: np.ndarray) -> float:
    """
    OLS upper bound (SFOLS Alg. 2): max w·v such that W v <= best, where W holds
    the explored weights and best the top scalarized value at each of them. The
    extrema in W keep it bounded. Returns -inf if the solver fails, so that a
    numerical failure skips this weight instead of ending the run.
    """
    res = linprog(-np.asarray(w, dtype=np.float64), A_ub=weights, b_ub=best,
                  bounds=(None, None), method="highs")
    return -res.fun if res.success else -np.inf


class GPILinearSupport:
    """Weight support M of GPI-LS (Alg. 1). Starts from the extrema weights."""
    def __init__(self, n_obj: int = 3, top_k: int = 2, max_support: int = 20,
                 max_corners: int = 20, min_distance: float = 0.05):
        self.n_obj       = n_obj
        self.top_k       = top_k
        self.max_support = max_support
        self.max_corners = max_corners
        self.min_distance = min_distance
        self.weights: List[np.ndarray] = list(np.eye(n_obj, dtype=np.float32))
        self.explored: List[np.ndarray] = list(self.weights)   # W_exp of SFOLS: dropped weights stay here

    def remove_dominated(self, values: np.ndarray) -> np.ndarray:
        """
        Drop weights whose value is not the best at any corner weight, and
        return the values that remain. Extrema weights are always kept, so the
        ends of the front keep being trained.
        """
        corners = corner_weights(values)
        scalarized = corners @ values.T                    # (n_corners, n_values)
        is_best = scalarized >= scalarized.max(axis=1, keepdims=True) - 1e-6
        keep = is_best.any(axis=0)
        keep[:self.n_obj] = True
        self.weights = [w for w, k in zip(self.weights, keep) if k]
        return values[keep]

    def candidates(self, values: np.ndarray) -> np.ndarray:
        """Corner weights of V that are not already in the support."""
        corners = corner_weights(values)
        support = np.array(self.weights)
        dist = np.linalg.norm(corners[:, None, :] - support[None, :, :], axis=2).min(axis=1)
        corners = corners[dist > self.min_distance]
        if len(corners) > self.max_corners:
            corners = corners[np.random.choice(len(corners), self.max_corners, replace=False)]
        return corners.astype(np.float32)


# ============================================================
# Dynamics Model (GPI-PD)
# ============================================================

class EnsembleLinear(nn.Module):
    """n_members independent linear layers, applied in one tensor operation."""
    def __init__(self, n_members: int, in_dim: int, out_dim: int):
        super().__init__()
        bound = 1.0 / np.sqrt(in_dim)    # Same initialization as nn.Linear
        self.weight = nn.Parameter(torch.empty(n_members, in_dim, out_dim).uniform_(-bound, bound))
        self.bias   = nn.Parameter(torch.empty(n_members, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight) + self.bias   # x: (M, B, in_dim)


class DynamicsEnsemble:
    """
    Ensemble of probabilistic MLPs predicting (obs_next - obs, r_vec) from
    (obs, action) as a diagonal Gaussian (Eq. 13-14 of GPI-PD). All members run
    as one batched network, and each one still trains on its own random batch.
    """
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int, n_members: int = 5,
                 hidden: int = 200, lr: float = 1e-3):
        self.obs_dim = obs_dim
        self.n_members = n_members
        out = obs_dim + n_obj
        self.net = nn.Sequential(
            EnsembleLinear(n_members, obs_dim + act_dim, hidden), nn.SiLU(),
            EnsembleLinear(n_members, hidden, hidden), nn.SiLU(),
            EnsembleLinear(n_members, hidden, hidden), nn.SiLU(),
            EnsembleLinear(n_members, hidden, hidden), nn.SiLU(),
            EnsembleLinear(n_members, hidden, 2 * out),
        ).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _gaussian_nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum over members of each member's mean NLL, so no member's gradient is scaled down."""
        mu, logvar = self.net(x).chunk(2, dim=-1)
        logvar = logvar.clamp(-10.0, 4.0)
        return 0.5 * self.n_members * (((mu - y) ** 2) * torch.exp(-logvar) + logvar).mean()

    def _batch(self, buffer: ReplayBuffer, idx: np.ndarray):
        """Normalized (x, y) tensors for an index array of shape (M, B)."""
        def t(arr): return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        x = t(np.concatenate([buffer.obs[idx], buffer.action[idx]], axis=-1))
        y = t(np.concatenate([buffer.obs_next[idx] - buffer.obs[idx], buffer.r_vec[idx]], axis=-1))
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def fit(self, buffer: ReplayBuffer, max_steps: int = 200, batch_size: int = 256,
            patience: int = 5, check_every: int = 10) -> float:
        """Train on the buffer with early stopping on a holdout set. Returns holdout NLL."""
        def t(arr): return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)
        n = len(buffer)
        # Normalization stats from a sample: a full pass over a 500k buffer is slow
        idx = np.random.randint(0, n, min(n, 50_000))
        x = np.concatenate([buffer.obs[idx], buffer.action[idx]], axis=1)
        y = np.concatenate([buffer.obs_next[idx] - buffer.obs[idx], buffer.r_vec[idx]], axis=1)
        self.x_mean, self.x_std = t(x.mean(0)), t(x.std(0) + 1e-6)
        self.y_mean, self.y_std = t(y.mean(0)), t(y.std(0) + 1e-6)
        self.y_min,  self.y_max = t(y.min(0)),  t(y.max(0))

        perm    = np.random.permutation(n)
        n_val   = min(max(n // 10, 1), 2000)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        x_val, y_val = self._batch(buffer, np.tile(val_idx, (self.n_members, 1)))

        best, since_best = float("inf"), 0
        for step in range(max_steps):
            idx = train_idx[np.random.randint(0, len(train_idx), (self.n_members, batch_size))]
            loss = self._gaussian_nll(*self._batch(buffer, idx))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if (step + 1) % check_every == 0:
                with torch.no_grad():
                    val = self._gaussian_nll(x_val, y_val).item() / self.n_members
                if val < best - 1e-3:
                    best, since_best = val, 0
                else:
                    since_best += 1
                    if since_best >= patience:
                        break
        return best

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, action: torch.Tensor):
        """Sample (obs_next, r_vec) from a random ensemble member per transition."""
        B = obs.shape[0]
        x = (torch.cat([obs, action], dim=-1) - self.x_mean) / self.x_std
        out = self.net(x.expand(self.n_members, -1, -1))        # (M, B, 2*out)
        pick = torch.randint(self.n_members, (B,), device=DEVICE)
        mu, logvar = out[pick, torch.arange(B, device=DEVICE)].chunk(2, dim=-1)
        y = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar.clamp(-10.0, 4.0))
        y = y * self.y_std + self.y_mean
        y = torch.max(torch.min(y, self.y_max), self.y_min)
        return obs + y[:, :self.obs_dim], y[:, self.obs_dim:]


# ============================================================
# Main Agent: MOMA_AC_Agent
# ============================================================

class MOMA_AC_Agent:
    """
    MOMA-AC: preference-conditioned TD3 voi vector-valued critic.
      omega_sampling='random' -> Dirichlet moi episode (baseline MOMA-AC goc)
      omega_sampling='ols'    -> Optimistic Linear Support: omega ~ M, hanh dong theo actor, khong dung Q de chon
      omega_sampling='gpi_ls' -> GPI Linear Support: omega ~ M, hanh dong theo GPI
      omega_sampling='gpi_pd' -> GPI-LS + GPI-prioritized replay + Dyna (hoc model)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_objectives: int       = 3,
        hidden_dim: int         = 256,
        actor_lr: float         = 3e-4,
        critic_lr: float        = 3e-4,
        gamma: float            = 0.99,
        tau: float              = 0.005,
        policy_noise: float     = 0.2,
        noise_clip: float       = 0.5,
        policy_delay: int       = 2,
        expl_noise: float       = 0.3,
        expl_noise_min: float   = 0.05,
        expl_noise_decay: float = 0.9999,
        buffer_size: int        = 500_000,
        omega_sampling: str     = "random",
        corner_warmup_episodes: int = 0,
        # GPI-LS
        gpi_top_k: int          = 2,
        gpi_max_support: int    = 20,
        gpi_max_corners: int    = 20,
        # GPI-PD
        per_alpha: float        = 0.6,
        per_kappa: float        = 0.1,
        dyna_beta: float        = 0.5,
        dyna_rollouts: int      = 2000,
        model_fit_steps: int    = 200,
        model_buffer_size: int  = 100_000,
    ):
        if omega_sampling not in ("random", "ols", "gpi_ls", "gpi_pd"):
            raise ValueError(f"Unknown omega_sampling: {omega_sampling}")

        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.n_obj     = n_objectives
        self.gamma     = gamma
        self.tau       = tau
        self.policy_noise    = policy_noise
        self.noise_clip      = noise_clip
        self.policy_delay    = policy_delay
        self.expl_noise      = expl_noise
        self.expl_noise_min  = expl_noise_min
        self.expl_noise_decay = expl_noise_decay
        self.omega_sampling  = omega_sampling
        self.corner_warmup_episodes = corner_warmup_episodes
        self.dyna_beta       = dyna_beta
        self.dyna_rollouts   = dyna_rollouts
        self.model_fit_steps = model_fit_steps
        self.per_kappa       = per_kappa

        # ── Networks ─────────────────────────────────────────────
        self.actor        = PreferencedActor(obs_dim, act_dim, n_objectives, hidden_dim).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1        = VectorCritic(obs_dim, act_dim, n_objectives, hidden_dim).to(DEVICE)
        self.critic2        = VectorCritic(obs_dim, act_dim, n_objectives, hidden_dim).to(DEVICE)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_opt     = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr)

        # ── Weight support (OLS, GPI-LS) ─────────────────────────
        self.use_gpi = omega_sampling in ("gpi_ls", "gpi_pd")
        self.support = (GPILinearSupport(n_objectives, gpi_top_k, gpi_max_support, gpi_max_corners)
                        if omega_sampling in ("ols", "gpi_ls", "gpi_pd") else None)

        # ── Replay Buffer / Dyna model ───────────────────────────
        self.model = None
        self.model_buffer = None
        if omega_sampling == "gpi_pd":
            self.buffer = PrioritizedReplayBuffer(obs_dim, act_dim, n_objectives, buffer_size,
                                                  per_alpha, per_kappa)
            self.model = DynamicsEnsemble(obs_dim, act_dim, n_objectives)
            self.model_buffer = ReplayBuffer(obs_dim, act_dim, n_objectives, model_buffer_size)
        else:
            self.buffer = ReplayBuffer(obs_dim, act_dim, n_objectives, buffer_size)

        # ── State ────────────────────────────────────────────────
        self.total_updates  = 0
        self.episode_count  = 0
        self.current_omega  = np.ones(n_objectives, dtype=np.float32) / n_objectives

        print(f"\n{'='*52}")
        print(f"MOMA-AC Agent  (TD3 backbone)")
        print(f"{'='*52}")
        print(f"  obs_dim        : {obs_dim}")
        print(f"  act_dim        : {act_dim}")
        print(f"  n_objectives   : {n_objectives}")
        print(f"  omega_sampling : {omega_sampling}")
        print(f"  device         : {DEVICE}")
        print(f"{'='*52}\n")

    # ── Omega Sampling ───────────────────────────────────────────

    def sample_omega(self) -> np.ndarray:
        """Call at start of each new episode."""
        if self.support is not None:
            # GPI-PD Alg. 2, line 10: w_t ~ M
            omega = self.support.weights[np.random.randint(len(self.support.weights))]
        elif self.episode_count < self.corner_warmup_episodes:
            # Random: corner warmup then Dirichlet
            omega = np.zeros(self.n_obj, dtype=np.float32)
            omega[self.episode_count % self.n_obj] = 1.0
        else:
            omega = np.random.dirichlet(np.ones(self.n_obj)).astype(np.float32)

        self.current_omega = omega
        self.episode_count += 1
        return omega

    def update_weight_support(self, env, n_episodes: int = 3, steps: int = 60) -> dict:
        """
        One iteration on the support M (GPI-PD Alg. 2, lines 5-8). GPI variants rank
        corner weights by measured GPI gain (Eq. 7), OLS by its optimistic bound.
        Both use rollout values only, so critic estimates don't enter either ranking.
        """
        values = np.array([self.eval_omega(env, w, n_episodes, steps)
                           for w in self.support.weights])
        values = self.support.remove_dominated(values)

        corners = self.support.candidates(values)
        gaps = np.zeros(len(corners))
        if self.use_gpi:
            for i, w in enumerate(corners):
                v_gpi   = self.eval_omega(env, w, n_episodes, steps, use_gpi=True)
                gaps[i] = v_gpi @ w - (values @ w).max()           # Eq. 7
        else:
            W = np.array(self.support.explored, dtype=np.float64)
            best = (W @ values.T).max(axis=1)
            for i, w in enumerate(corners):
                gaps[i] = optimistic_value(w, W, best) - (values @ w).max()
            gaps[gaps < 1e-6] = 0.0                                # Solver noise, not a real gap

        free = self.support.max_support - len(self.support.weights)
        n_add = max(min(self.support.top_k, free), 0)
        added = 0
        for i in np.argsort(-gaps)[:n_add]:
            if gaps[i] > 0:     # Eq. 7 is noisy; a corner with no measured gain isn't worth training
                self.support.weights.append(corners[i].copy())
                self.support.explored.append(corners[i].copy())
                added += 1

        return {
            "support_size": len(self.support.weights),
            "n_corners":    len(corners),
            "max_gap":      float(gaps.max()) if len(gaps) else 0.0,
            "added":        added,
        }

    # ── Action Selection ─────────────────────────────────────────

    def _scalarized_q(self, critic_a, critic_b, obs, action, cond_omega, omega) -> torch.Tensor:
        """min(Q_a, Q_b) · omega for critics conditioned on cond_omega. Shape (B,)."""
        q_a = (critic_a(obs, action, cond_omega) * omega).sum(-1)
        q_b = (critic_b(obs, action, cond_omega) * omega).sum(-1)
        return torch.min(q_a, q_b)

    def _support_tensor(self, extra: Optional[np.ndarray] = None) -> torch.Tensor:
        weights = list(self.support.weights)
        if extra is not None:
            weights.append(extra)
        return torch.as_tensor(np.array(weights), dtype=torch.float32, device=DEVICE)

    @torch.no_grad()
    def gpi_select_action(self, obs: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        GPI policy of a weight-conditioned network (GPI-PD Eq. 21):
        argmax over a in {pi(s, w')} of max over w' in M of Q(s, a, w') · omega.
        omega itself joins the set, so GPI can always fall back on pi(s, omega).
        """
        M = self._support_tensor(omega)
        K = M.shape[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=DEVICE)

        candidates = self.actor(obs_t.repeat(K, 1), M)                     # (K, act_dim)
        q = self._scalarized_q(self.critic1, self.critic2,
                               obs_t.repeat(K * K, 1),
                               candidates.repeat_interleave(K, dim=0),
                               M.repeat(K, 1),
                               omega_t)
        best = q.view(K, K).max(dim=1).values.argmax()
        return candidates[best].cpu().numpy()

    @torch.no_grad()
    def _gpi_action_batch(self, obs: torch.Tensor, omega: np.ndarray) -> torch.Tensor:
        """
        Batched GPI action for Dyna rollouts. Each candidate pi(s, w') is scored
        only with its own Q(s, a, w'), since the full K x K sweep is too slow per batch.
        """
        M = self._support_tensor(omega)
        B, K = obs.shape[0], M.shape[0]
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=DEVICE)

        obs_r = obs.repeat_interleave(K, dim=0)
        cond  = M.repeat(B, 1)
        a     = self.actor(obs_r, cond)
        q     = self._scalarized_q(self.critic1, self.critic2, obs_r, a, cond, omega_t)
        best  = q.view(B, K).argmax(dim=1)
        return a.view(B, K, -1)[torch.arange(B, device=DEVICE), best]

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, omega: np.ndarray,
                      explore: bool = True, use_gpi: bool = False) -> np.ndarray:
        if use_gpi:
            action = self.gpi_select_action(obs, omega)
        else:
            obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            omega_t = torch.FloatTensor(omega).unsqueeze(0).to(DEVICE)
            action  = self.actor(obs_t, omega_t).cpu().numpy()[0]
        if explore:
            n_pose = self.actor.n_pose_dims
            noise = np.random.normal(0, self.expl_noise, size=action.shape)
            # Clip per range: position dims [-1,1], power and BW dims [0,1]
            action[:n_pose] = np.clip(action[:n_pose] + noise[:n_pose], -1.0, 1.0)
            action[n_pose:] = np.clip(action[n_pose:] + noise[n_pose:],  0.0, 1.0)
            self.expl_noise = max(self.expl_noise_min,
                                  self.expl_noise * self.expl_noise_decay)
        return action

    # ── Dyna (GPI-PD) ────────────────────────────────────────────

    def dyna_step(self) -> dict:
        """Fit the model, then add GPI-prioritized simulated transitions (Alg. 2, lines 15-19)."""
        val_nll = self.model.fit(self.buffer, max_steps=self.model_fit_steps)

        idx   = self.buffer.sample_indices(self.dyna_rollouts)
        obs_t = torch.FloatTensor(self.buffer.obs[idx]).to(DEVICE)
        action   = self._gpi_action_batch(obs_t, self.current_omega)
        obs_next, r_vec = self.model.predict(obs_t, action)
        omega = np.tile(self.current_omega, (self.dyna_rollouts, 1))
        self.model_buffer.add_batch(self.buffer.obs[idx], action.cpu().numpy(),
                                    r_vec.cpu().numpy(), obs_next.cpu().numpy(), omega)
        return {"model_val_nll": val_nll}

    @torch.no_grad()
    def _gpi_td_errors(self, obs, action, r_vec, obs_next, done) -> np.ndarray:
        """Eq. 10 of GPI-PD for the current omega, scoring each pi(s', w') with its own Q."""
        omega = self.current_omega
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=DEVICE)
        M = self._support_tensor(omega)
        B, K = obs.shape[0], M.shape[0]

        obs_n_r = obs_next.repeat_interleave(K, dim=0)
        cond    = M.repeat(B, 1)
        a_next  = self.actor_target(obs_n_r, cond)
        q_next  = self._scalarized_q(self.critic1_target, self.critic2_target,
                                     obs_n_r, a_next, cond, omega_t)
        q_next  = q_next.view(B, K).max(dim=1).values

        omega_b = omega_t.expand(B, -1)
        q = (self.critic1(obs, action, omega_b) * omega_t).sum(-1)
        target = (r_vec * omega_t).sum(-1) + self.gamma * (1.0 - done[:, 0]) * q_next
        return (target - q).cpu().numpy()

    # ── TD3 Update ───────────────────────────────────────────────

    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}

        # GPI-PD Alg. 2, line 21: beta*b simulated tuples and (1-beta)*b real tuples
        n_model = 0
        if self.model_buffer is not None and len(self.model_buffer) >= batch_size:
            n_model = int(self.dyna_beta * batch_size)
        idx  = self.buffer.sample_indices(batch_size - n_model)
        real = self.buffer.get(idx)
        obs_b, act_b, r_b, obs_n_b, omega_b, done_b = real
        if n_model > 0:
            fake = self.model_buffer.get(self.model_buffer.sample_indices(n_model))
            obs_b, act_b, r_b, obs_n_b, omega_b, done_b = (
                torch.cat(pair, dim=0) for pair in zip(real, fake))

        # GPI-PD Alg. 2, line 22: L(theta; w_t) + L(theta; w'), w' ~ M, so that
        # training on one weight doesn't make the network forget the others
        if self.support is not None:
            n = obs_b.shape[0]
            M = self._support_tensor()
            w_t = torch.as_tensor(self.current_omega, dtype=torch.float32, device=DEVICE)
            omega_b = torch.cat([w_t.expand(n, -1),
                                 M[torch.randint(M.shape[0], (n,), device=DEVICE)]], dim=0)
            obs_b, act_b, r_b, obs_n_b, done_b = (
                x.repeat(2, 1) for x in (obs_b, act_b, r_b, obs_n_b, done_b))

        with torch.no_grad():
            # Target policy smoothing (TD3) — clamp per range
            noise  = (torch.randn_like(act_b) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip)
            next_a_raw = self.actor_target(obs_n_b, omega_b) + noise
            # Position dims clamp [-1,1], power and BW dims clamp [0,1]
            n_pose = self.actor.n_pose_dims
            next_a = torch.cat([
                next_a_raw[..., :n_pose].clamp(-1., 1.),
                next_a_raw[..., n_pose:].clamp(0.,  1.),
            ], dim=-1)

            # Twin critic targets — pick the critic with the lower scalarized value
            # (MOMA-AC Eq. 4-6, GPI-PD appendix Eq. 19)
            Q1_next = self.critic1_target(obs_n_b, next_a, omega_b)  # (B, n_obj)
            Q2_next = self.critic2_target(obs_n_b, next_a, omega_b)
            use_q1  = (Q1_next * omega_b).sum(-1, keepdim=True) <= (Q2_next * omega_b).sum(-1, keepdim=True)
            Q_next  = torch.where(use_q1, Q1_next, Q2_next)          # (B, n_obj)

            # Bellman target (vector-valued, NO scalarization here)
            target = r_b + self.gamma * (1.0 - done_b) * Q_next   # (B, n_obj)

        # Critic loss: VECTOR — train all objectives every transition
        # (Scalarizing here would starve objectives not in current omega)
        Q1 = self.critic1(obs_b, act_b, omega_b)
        Q2 = self.critic2(obs_b, act_b, omega_b)
        if self.model is not None:
            # Prioritized sampling is paired with a Huber loss (LAP, Fujimoto et al. 2020)
            c_loss = (F.smooth_l1_loss(Q1, target, beta=self.per_kappa)
                      + F.smooth_l1_loss(Q2, target, beta=self.per_kappa))
        else:
            c_loss = F.mse_loss(Q1, target) + F.mse_loss(Q2, target)

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 10.)
        self.critic_opt.step()
        self.total_updates += 1

        info = {"critic_loss": float(c_loss)}

        # GPI-PD Alg. 2, line 23
        if self.model is not None:
            r_obs, r_act, r_rew, r_next, _, r_done = real
            self.buffer.update_priorities(idx, self._gpi_td_errors(r_obs, r_act, r_rew, r_next, r_done))

        # Delayed actor update (every policy_delay critic steps)
        if self.total_updates % self.policy_delay == 0:
            Q_a    = self.critic1(obs_b, self.actor(obs_b, omega_b), omega_b)
            a_loss = -(Q_a * omega_b).sum(-1).mean()
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_opt.step()

            # Soft update target networks
            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            info["actor_loss"] = float(a_loss)

        return info

    def _soft_update(self, net: nn.Module, target: nn.Module):
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)

    # ── Pareto Evaluation ────────────────────────────────────────

    @torch.no_grad()
    def eval_omega(self, env, omega: np.ndarray, n_episodes: int = 5, steps: int = 60,
                   use_gpi: bool = False) -> np.ndarray:
        """Evaluate policy for one omega. Returns mean reward vector."""
        rng_state = np.random.get_state()
        uid    = sorted(env.uavs.keys())[0]
        all_rv = []
        all_sla = []
        for ep in range(n_episodes):
            np.random.seed(EVAL_SEED + ep)
            obs_dict = env.reset()
            obs      = obs_dict[uid]
            ep_rv    = []
            ep_sla   = []
            done     = False
            t        = 0
            while not done and t < steps:
                action   = self.select_action(obs, omega, explore=False, use_gpi=use_gpi)
                obs_dict, _, done, info = env.step({uid: action})
                obs = obs_dict.get(uid, obs)
                rv  = info.get("reward_vector", None)
                if rv is not None:
                    ep_rv.append(np.array(rv, dtype=np.float32))
                ep_sla.append(sla_of(info))
                t += 1
            if ep_rv:
                all_rv.append(np.mean(ep_rv, axis=0))
                all_sla.append(nan_mean(ep_sla))
        np.random.set_state(rng_state)
        self.last_sla = nan_mean(all_sla)   # Per slice: mean over episodes of the episode's mean SLA share
        return np.mean(all_rv, axis=0) if all_rv else np.zeros(self.n_obj, dtype=np.float32)

    def eval_pareto_front(self, env, n_omega: int = 30,
                          n_episodes: int = 5, steps: int = 60,
                          use_gpi: bool = False):
        """Sweep a fixed set of n_omega weights, eval each. use_gpi=True uses GPI inference."""
        corners  = np.eye(self.n_obj, dtype=np.float32)
        interior = np.random.RandomState(EVAL_SEED).dirichlet(
            np.ones(self.n_obj), n_omega - self.n_obj).astype(np.float32)
        omegas   = np.vstack([corners, interior])
        results  = []
        self.sweep_sla = []   # SLA shares per omega, read by the trainer
        for i, w in enumerate(omegas):
            rv = self.eval_omega(env, w, n_episodes, steps, use_gpi=use_gpi)
            results.append(rv)
            self.sweep_sla.append(self.last_sla)
            print(f"  eval {i+1}/{len(omegas)}: w={np.round(w,2)} " +
                  (f"QoS={rv[0]:.3f} Eng={rv[1]:.3f} Fair={rv[2]:.3f}" if self.n_obj == 3 else f"r={np.round(rv, 3)}"))
        return np.array(results), omegas

    # ── Save / Load ──────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic_opt":     self.critic_opt.state_dict(),
            # Dynamics model (gpi_pd mode)
            "model_state":    self.model.net.state_dict() if self.model else None,
            "model_opt":      self.model.opt.state_dict() if self.model else None,
            "expl_noise":     self.expl_noise,
            "total_updates":  self.total_updates,
            "episode_count":  self.episode_count,
            "omega_sampling": self.omega_sampling,
            "current_omega":  self.current_omega.tolist(),
            # GPI-LS weight support
            "support_weights": [w.tolist() for w in self.support.weights] if self.support else [],
            "explored_weights": [w.tolist() for w in self.support.explored] if self.support else [],
        }, path)
        if self.support:
            print(f"[MOMA-AC] Saved: {path}  (support={len(self.support.weights)} weights)")
        else:
            print(f"[MOMA-AC] Saved: {path}")

    def load(self, path: str):
        st = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(st["actor"])
        self.actor_target.load_state_dict(st["actor_target"])
        self.critic1.load_state_dict(st["critic1"])
        self.critic2.load_state_dict(st["critic2"])
        self.critic1_target.load_state_dict(st["critic1_target"])
        self.critic2_target.load_state_dict(st["critic2_target"])
        if "actor_opt" in st:
            self.actor_opt.load_state_dict(st["actor_opt"])
            self.critic_opt.load_state_dict(st["critic_opt"])
        if self.model and st.get("model_state"):
            self.model.net.load_state_dict(st["model_state"])
            self.model.opt.load_state_dict(st["model_opt"])
        self.expl_noise    = st.get("expl_noise",    self.expl_noise)
        self.total_updates = st.get("total_updates", 0)
        self.episode_count = st.get("episode_count", 0)
        self.current_omega = np.array(st.get("current_omega", self.current_omega),
                                      dtype=np.float32)
        if self.support and st.get("support_weights"):
            self.support.weights = [np.array(w, np.float32) for w in st["support_weights"]]
            self.support.explored = [np.array(w, np.float32)
                                     for w in st.get("explored_weights") or st["support_weights"]]
            print(f"[MOMA-AC] Loaded support: {len(self.support.weights)} weights")
        print(f"[MOMA-AC] Loaded: {path} "
              f"(ep={self.episode_count}, updates={self.total_updates})")


# ============================================================
# Several UAVs: centralised training, decentralised execution
# ============================================================

def episode_diagnostics(steps: list) -> dict:
    """Team diagnostics of one episode, from (coverage, handovers, min_uav_distance, backhaul_limited) per step."""
    coverage, handovers, distance, backhaul = np.array(steps, dtype=np.float64).T
    return {"coverage": float(coverage.mean()), "handovers": float(handovers.mean()),
            "backhaul_limited": float(backhaul.mean()),
            "min_uav_distance": float(distance.min()),    # closest approach of the episode
            "mean_uav_distance": float(distance.mean())}  # mean over steps of each step's closest pair


def average_diagnostics(episodes: list) -> dict:
    """Mean of episode means, as for the reward vector; min_uav_distance stays the true minimum."""
    return {key: min(ep[key] for ep in episodes) if key == "min_uav_distance"
                 else float(np.mean([ep[key] for ep in episodes]))
            for key in episodes[0]}


class MOMA_CTDE_Agent(MOMA_AC_Agent):
    """
    MOMA-TD3 of the paper (Alg. 1) for a team of UAVs sharing one reward vector.
      Actor   : pi_i(o_i, omega), uses only what UAV i observes
      Critics : Q(o, a, omega) over the joint observation and joint action
    obs_dim and act_dim are per UAV. Critics and buffer hold them flattened in UAV id
    order: one buffer is the paper's per-agent buffers sampled in sync by timestep.
    """

    def __init__(self, obs_dim: int, act_dim: int, n_agents: int, n_objectives: int = 3,
                 hidden_dim: int = 256, actor_lr: float = 3e-4,
                 omega_sampling: str = "random", omega_per_agent: float = 0.0,
                 omega_switch_prob: float = 0.0, network: str = "mlp", **kwargs):
        if omega_sampling != "random":
            raise ValueError(f"MOMA_CTDE_Agent only supports omega_sampling='random', got {omega_sampling}")

        super().__init__(n_agents * obs_dim, n_agents * act_dim, n_objectives, hidden_dim,
                         actor_lr=actor_lr, omega_sampling=omega_sampling, **kwargs)
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.n_agents = n_agents

        if network not in ("mlp", "attention"):
            raise ValueError(f"network must be mlp or attention, got {network}")
        self.network = network
        if network == "mlp":
            self.actor = MultiHeadActor(obs_dim, act_dim, n_agents, n_objectives, hidden_dim).to(DEVICE)
        else:
            # Thesis item B2: one actor for every UAV and an attention critic, both independent of N
            self.self_dim = obs_dim - N_OBS_OTHER * (n_agents - 1)
            self.actor = AttentionActor(self.self_dim, act_dim, n_objectives).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # One omega per UAV (thesis item B1). Off (0): one omega for the team, as in the paper. On: omega has
        # shape (n_agents, n_obj); an episode draws an independent Dirichlet omega for each UAV with probability
        # omega_per_agent, else one shared omega repeated. pi_i sees only its own omega_i, the critics see all
        # of them, and UAV i's actor loss is omega_i . Q_team: the reward stays one team vector, and with the
        # other UAVs' replay actions held fixed (Eq. 10-12) a UAV that weights energy mostly saves its own.
        # omega_switch_prob redraws omega within an episode, so the policy sees omega change at run time
        self.per_agent_omega   = omega_per_agent > 0
        self.omega_per_agent   = omega_per_agent
        self.omega_switch_prob = omega_switch_prob
        if self.per_agent_omega or network == "attention":
            w_dim = n_agents * n_objectives if self.per_agent_omega else n_objectives
            critic_lr = self.critic_opt.param_groups[0]["lr"]
            if network == "attention":
                self.critic1 = AttentionCritic(self.self_dim, act_dim, n_objectives).to(DEVICE)
                self.critic2 = AttentionCritic(self.self_dim, act_dim, n_objectives).to(DEVICE)
            else:
                self.critic1 = VectorCritic(n_agents * obs_dim, n_agents * act_dim, n_objectives, hidden_dim,
                                            omega_dim=w_dim).to(DEVICE)
                self.critic2 = VectorCritic(n_agents * obs_dim, n_agents * act_dim, n_objectives, hidden_dim,
                                            omega_dim=w_dim).to(DEVICE)
            self.critic1_target = copy.deepcopy(self.critic1)
            self.critic2_target = copy.deepcopy(self.critic2)
            self.critic_opt = torch.optim.Adam(
                list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
            self.buffer = ReplayBuffer(n_agents * obs_dim, n_agents * act_dim, n_objectives,
                                       self.buffer.max_size, omega_dim=w_dim)
            if self.per_agent_omega:
                self.current_omega = np.full((n_agents, n_objectives), 1.0 / n_objectives, dtype=np.float32)

        self.eval_diagnostics: list = []
        print(f"[MOMA-AC] CTDE: {n_agents} agents, obs_dim={obs_dim}, act_dim={act_dim} per agent\n")

    # ── Omega ────────────────────────────────────────────────────

    def _draw_team_omega(self) -> np.ndarray:
        if not self.per_agent_omega:
            return np.random.dirichlet(np.ones(self.n_obj)).astype(np.float32)
        if np.random.rand() < self.omega_per_agent:
            return np.random.dirichlet(np.ones(self.n_obj), self.n_agents).astype(np.float32)
        return np.tile(np.random.dirichlet(np.ones(self.n_obj)), (self.n_agents, 1)).astype(np.float32)

    def sample_omega(self) -> np.ndarray:
        """Call at the start of each episode. Without per-UAV omega this is the base class's draw."""
        if not self.per_agent_omega:
            return super().sample_omega()
        omega = self._draw_team_omega()
        self.current_omega = omega
        self.episode_count += 1
        return omega

    def maybe_switch_omega(self, omega: np.ndarray) -> np.ndarray:
        """Within an episode: a new omega with probability omega_switch_prob (no random draw when it is 0)"""
        if self.omega_switch_prob <= 0 or np.random.rand() >= self.omega_switch_prob:
            return omega
        self.current_omega = self._draw_team_omega()
        return self.current_omega

    def _omega_i(self, omega_b: torch.Tensor, i: int) -> torch.Tensor:
        """UAV i's omega in a batch of (team or per-UAV) omegas"""
        if not self.per_agent_omega:
            return omega_b
        return omega_b[:, i * self.n_obj:(i + 1) * self.n_obj]

    def _team_omega(self, omega_b: torch.Tensor) -> torch.Tensor:
        """The weight that picks the lower twin in the target: the mean of the UAVs' omegas"""
        if not self.per_agent_omega:
            return omega_b
        return omega_b.view(omega_b.shape[0], self.n_agents, self.n_obj).mean(dim=1)

    def _omega_matrix(self, omega_b: torch.Tensor) -> torch.Tensor:
        """(B, n_agents, n_obj): every UAV's omega, from a team omega or per-UAV omegas"""
        B = omega_b.shape[0]
        if self.per_agent_omega:
            return omega_b.view(B, self.n_agents, self.n_obj)
        return omega_b.unsqueeze(1).expand(B, self.n_agents, self.n_obj)

    def _act(self, net, obs_i: torch.Tensor, omega_b: torch.Tensor, i: int) -> torch.Tensor:
        """UAV i's action from actor net: the attention actor also sees the other UAVs' omegas"""
        if self.network == "attention":
            return net(obs_i, self._omega_matrix(omega_b), i)
        return net(obs_i, self._omega_i(omega_b, i), i)

    def critic_omega(self, omega: np.ndarray) -> np.ndarray:
        """Omega as the critics take it: (n_obj,) for a team omega, or the per-UAV omegas flattened"""
        omega = np.asarray(omega, dtype=np.float32).reshape(-1, self.n_obj)
        if not self.per_agent_omega:
            return omega[0]
        return np.broadcast_to(omega, (self.n_agents, self.n_obj)).reshape(-1)

    # ── Action Selection ─────────────────────────────────────────

    @torch.no_grad()
    def select_joint_action(self, obs: np.ndarray, omega: np.ndarray,
                            explore: bool = True) -> np.ndarray:
        """obs has shape (n_agents, obs_dim). Each UAV acts on its own obs_i and its omega only.
        omega is one team omega (n_obj,) or one per UAV (n_agents, n_obj)."""
        omega_t = torch.as_tensor(np.asarray(omega, dtype=np.float32).reshape(-1, self.n_obj),
                                  dtype=torch.float32, device=DEVICE)
        obs_t   = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=DEVICE)
        if self.network == "attention":
            omegas = omega_t.expand(self.n_agents, self.n_obj).unsqueeze(0)   # (1, N, n_obj)
            action = self.actor.forward_all(obs_t.unsqueeze(0), omegas).cpu().numpy()[0]
        else:
            action = np.stack([self.actor(obs_t[i:i + 1], omega_t[i:i + 1] if len(omega_t) > 1 else omega_t,
                                          i).cpu().numpy()[0]
                               for i in range(self.n_agents)])
        if explore:
            n_pose = self.actor.n_pose_dims
            noise = np.random.normal(0, self.expl_noise, size=action.shape)
            # Clip per range: position dims [-1,1], power and BW dims [0,1]
            action[:, :n_pose] = np.clip(action[:, :n_pose] + noise[:, :n_pose], -1.0, 1.0)
            action[:, n_pose:] = np.clip(action[:, n_pose:] + noise[:, n_pose:],  0.0, 1.0)
            self.expl_noise = max(self.expl_noise_min,
                                  self.expl_noise * self.expl_noise_decay)
        return action

    # ── Update ───────────────────────────────────────────────────

    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}

        idx = self.buffer.sample_indices(batch_size)
        obs_b, act_b, r_b, obs_n_b, omega_b, done_b = self.buffer.get(idx)
        B, n = obs_b.shape[0], self.n_agents
        obs_i   = obs_b.view(B, n, self.obs_dim)
        obs_n_i = obs_n_b.view(B, n, self.obs_dim)

        with torch.no_grad():
            # Joint next action: each head plus its own clipped noise (Eq. 1-2)
            n_pose = self.actor.n_pose_dims
            next_a = []
            if self.network == "attention":
                raw   = self.actor_target.forward_all(obs_n_i, self._omega_matrix(omega_b))
                noise = (torch.randn_like(raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                raw   = raw + noise
                next_a = [torch.cat([raw[..., :n_pose].clamp(-1., 1.), raw[..., n_pose:].clamp(0., 1.)],
                                    dim=-1).reshape(B, n * self.act_dim)]
            for i in range(n if self.network != "attention" else 0):
                raw   = self._act(self.actor_target, obs_n_i[:, i], omega_b, i)
                noise = (torch.randn_like(raw) * self.policy_noise).clamp(
                            -self.noise_clip, self.noise_clip)
                raw   = raw + noise
                next_a.append(torch.cat([raw[..., :n_pose].clamp(-1., 1.),
                                         raw[..., n_pose:].clamp(0.,  1.)], dim=-1))
            next_a = torch.cat(next_a, dim=-1)

            # Target from the critic with the lower scalarized value (Eq. 4-6)
            Q1_next = self.critic1_target(obs_n_b, next_a, omega_b)
            Q2_next = self.critic2_target(obs_n_b, next_a, omega_b)
            w_team  = self._team_omega(omega_b)
            use_q1  = (Q1_next * w_team).sum(-1, keepdim=True) <= (Q2_next * w_team).sum(-1, keepdim=True)
            Q_next  = torch.where(use_q1, Q1_next, Q2_next)
            target  = r_b + self.gamma * (1.0 - done_b) * Q_next

        # Critic loss (Eq. 8-9)
        c_loss = (F.mse_loss(self.critic1(obs_b, act_b, omega_b), target)
                  + F.mse_loss(self.critic2(obs_b, act_b, omega_b), target))
        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 10.)
        self.critic_opt.step()
        self.total_updates += 1

        info = {"critic_loss": float(c_loss)}

        # Delayed actor update (every policy_delay critic steps)
        if self.total_updates % self.policy_delay == 0:
            # Eq. 10-12: in copy i of the batch only a_i comes from the actor, the
            # other UAVs keep their replay actions. One critic pass over all copies.
            # With one omega per UAV, copy i is scalarized by UAV i's own omega_i
            joint = act_b.repeat(n, 1)
            own_a = (self.actor.forward_all(obs_i, self._omega_matrix(omega_b))
                     if self.network == "attention" else None)
            for i in range(n):
                start = i * self.act_dim
                joint[i * B:(i + 1) * B, start:start + self.act_dim] = (
                    own_a[:, i] if own_a is not None else self._act(self.actor, obs_i[:, i], omega_b, i))
            omega_r = omega_b.repeat(n, 1)
            Q_a     = self.critic1(obs_b.repeat(n, 1), joint, omega_r)
            w_loss  = torch.cat([self._omega_i(omega_b, i) for i in range(n)], dim=0)
            a_loss  = -(Q_a * w_loss).sum(-1).view(n, B).mean(dim=1).sum()
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_opt.step()

            # Soft update target networks
            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            info["actor_loss"] = float(a_loss)

        return info

    # ── Pareto Evaluation ────────────────────────────────────────

    @torch.no_grad()
    def eval_omega(self, env, omega: np.ndarray, n_episodes: int = 5, steps: int = 60,
                   use_gpi: bool = False) -> np.ndarray:
        """Evaluate the team for one omega, each UAV acting on its own observation."""
        if use_gpi:
            raise ValueError("MOMA_CTDE_Agent has no GPI policy")
        rng_state = np.random.get_state()
        uids   = sorted(env.uavs.keys())
        all_rv = []
        all_sla = []
        for ep in range(n_episodes):
            np.random.seed(EVAL_SEED + ep)
            obs_dict = env.reset()
            ep_rv    = []
            ep_diag  = []
            ep_sla   = []
            done     = False
            t        = 0
            while not done and t < steps:
                action = self.select_joint_action(np.stack([obs_dict[u] for u in uids]), omega, explore=False)
                obs_dict, _, done, info = env.step({u: action[i] for i, u in enumerate(uids)})
                ep_rv.append(np.array(info["reward_vector"], dtype=np.float32))
                ep_diag.append((info["coverage"], info["handovers"], info["min_uav_distance"], info["backhaul_limited"]))
                ep_sla.append(sla_of(info))
                t += 1
            if ep_rv:
                all_rv.append(np.mean(ep_rv, axis=0))
                all_sla.append(nan_mean(ep_sla))
                self.eval_diagnostics.append(episode_diagnostics(ep_diag))
        np.random.set_state(rng_state)
        self.last_sla = nan_mean(all_sla)
        return np.mean(all_rv, axis=0) if all_rv else np.zeros(self.n_obj, dtype=np.float32)

    def pop_eval_diagnostics(self) -> dict:
        """Team diagnostics over the eval episodes since the last call."""
        result = average_diagnostics(self.eval_diagnostics)
        self.eval_diagnostics = []
        return result
