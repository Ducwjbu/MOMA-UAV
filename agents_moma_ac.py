"""
agents_moma_ac.py
=================
MOMA-AC: Multi-Objective Actor-Critic voi Preference Conditioning (TD3 backbone)
- PreferencedActor : obs || omega concatenation → split sigmoid/tanh output
- VectorCritic     : twin critics output R^n_obj
- ReplayBuffer     : luu omega theo tung transition
- SimpleOLS        : standalone OLS weight selector
- MOMA_AC_Agent    : main class, omega_sampling='random'|'ols'
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Actor  (Concatenation conditioning — obs || omega → MLP)
# ============================================================

class PreferencedActor(nn.Module):
    """
    Preference-conditioned actor: concatenate omega with obs as input.
    Split output: tanh for pose dims, sigmoid for BW dims.
    """
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int = 3,
                 hidden: int = 256, n_pose_dims: int = 4):
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
        pose = torch.tanh(raw[..., :self.n_pose_dims])   # pos+power: [-1, 1]
        bw   = torch.sigmoid(raw[..., self.n_pose_dims:]) # BW alloc:  [ 0, 1]
        return torch.cat([pose, bw], dim=-1)


# ============================================================
# Critic  (concatenation conditioning, output R^n_obj)
# ============================================================

class VectorCritic(nn.Module):
    """Q(obs, action, omega) -> R^n_obj (one Q-value per objective)."""
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int = 3, hidden: int = 256):
        super().__init__()
        inp = obs_dim + act_dim + n_obj
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
# Replay Buffer  (stores omega per transition)
# ============================================================

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, n_obj: int,
                 max_size: int = 500_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs      = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.action   = np.zeros((max_size, act_dim),  dtype=np.float32)
        self.r_vec    = np.zeros((max_size, n_obj),    dtype=np.float32)
        self.obs_next = np.zeros((max_size, obs_dim),  dtype=np.float32)
        self.omega    = np.zeros((max_size, n_obj),    dtype=np.float32)
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

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        def t(arr): return torch.FloatTensor(arr[idx]).to(DEVICE)
        return t(self.obs), t(self.action), t(self.r_vec), \
               t(self.obs_next), t(self.omega), t(self.done)

    def __len__(self):
        return self.size


# ============================================================
# Policy Bank  (actor snapshots for GPI inference)
# ============================================================

class PolicyBank:
    """
    Luu toi da max_policies snapshot cua actor sau moi OLS omega.
    GPI inference: chon action tu policy co scalarized Q-value cao nhat.
    """
    def __init__(self, max_policies: int = 10):
        self.max_policies = max_policies
        self.actor_states: List[dict]       = []  # state_dicts
        self.omegas:        List[np.ndarray] = []  # omega tương ứng

    def add(self, actor: nn.Module, omega: np.ndarray):
        if len(self.actor_states) < self.max_policies:
            self.actor_states.append(copy.deepcopy(actor.state_dict()))
            self.omegas.append(omega.copy())
            print(f"[Bank] Added policy {len(self.actor_states)}/{self.max_policies} "
                  f"for ω={np.round(omega, 2)}")

    def __len__(self):
        return len(self.actor_states)



class SimpleOLS:
    """
    Phase 1 (corner): Xoay vong qua n_obj corner weights [1,0,0], [0,1,0], [0,0,1].
    Phase 2 (ols)   : Chon omega co max gap (ideal-point upper bound - current best).
    """
    def __init__(self, n_obj: int = 3, n_candidates: int = 500,
                 epsilon: float = 0.005):
        self.n_obj        = n_obj
        self.n_candidates = n_candidates
        self.epsilon      = epsilon
        self.corner_idx   = 0
        self.phase        = "corner"
        self.omegas: List[np.ndarray] = []
        self.values: List[np.ndarray] = []

    def update(self, omega: np.ndarray, value_vec: np.ndarray):
        self.omegas.append(omega.copy())
        self.values.append(value_vec.copy())
        if self.phase == "corner" and self.corner_idx >= self.n_obj:
            self.phase = "ols"
            print("[SimpleOLS] Corner phase done -> OLS phase")

    def select_next(self) -> np.ndarray:
        # Phase 1: corner weights
        if self.phase == "corner":
            w = np.zeros(self.n_obj, dtype=np.float32)
            w[self.corner_idx % self.n_obj] = 1.0
            self.corner_idx += 1
            return w

        if not self.values:
            return np.random.dirichlet(np.ones(self.n_obj)).astype(np.float32)

        values = np.array(self.values, dtype=np.float64)  # (N, n_obj)

        # Use 90th-percentile as ideal point instead of absolute max
        # This prevents early bad episodes (e.g. Energy=-0.8) from skewing UB
        ideal = np.percentile(values, 90, axis=0)

        # Sample candidate omegas uniformly on simplex
        cands    = np.random.dirichlet(np.ones(self.n_obj), self.n_candidates)
        ubs      = cands @ ideal                           # (n_cands,) upper bound
        cur_best = (cands @ values.T).max(axis=1)         # (n_cands,) current best
        gaps     = ubs - cur_best
        best_i   = int(gaps.argmax())

        if float(gaps[best_i]) < self.epsilon:
            # Converged: random fallback
            return np.random.dirichlet(np.ones(self.n_obj)).astype(np.float32)
        return cands[best_i].astype(np.float32)


# ============================================================
# Main Agent: MOMA_AC_Agent
# ============================================================

class MOMA_AC_Agent:
    """
    MOMA-AC: preference-conditioned TD3 voi vector-valued critic.
      omega_sampling='random' -> Dirichlet moi episode
      omega_sampling='ols'    -> OLS gap-maximizing moi episode
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
        corner_warmup_episodes: int = 30,
    ):
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

        # ── Replay Buffer ────────────────────────────────────────
        self.buffer = ReplayBuffer(obs_dim, act_dim, n_objectives, buffer_size)

        # ── OLS / Policy Bank ────────────────────────────────────
        self.ols  = SimpleOLS(n_objectives) if omega_sampling in ("ols", "gpi_ols") else None
        self.bank = PolicyBank(max_policies=10) if omega_sampling == "gpi_ols" else None
        # Track previous omega to detect OLS transitions
        self._prev_ols_omega: Optional[np.ndarray] = None

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

    def sample_omega(self, episode_value_vec: Optional[np.ndarray]) -> np.ndarray:
        """Call at start of each new episode. Pass last episode's mean r_vec."""
        if self.omega_sampling in ("ols", "gpi_ols"):
            if episode_value_vec is not None:
                self.ols.update(self.current_omega, episode_value_vec)
            new_omega = self.ols.select_next()

            # GPI_OLS: snapshot actor when OLS transitions to a meaningfully different omega
            if (self.omega_sampling == "gpi_ols"
                    and self._prev_ols_omega is not None
                    and episode_value_vec is not None
                    and not np.allclose(new_omega, self._prev_ols_omega, atol=0.08)
                    and len(self.bank) < self.bank.max_policies):
                self.bank.add(self.actor, self._prev_ols_omega)

            self._prev_ols_omega = new_omega.copy()
            omega = new_omega
        else:
            # Random: corner warmup then Dirichlet
            if self.episode_count < self.corner_warmup_episodes:
                omega = np.zeros(self.n_obj, dtype=np.float32)
                omega[self.episode_count % self.n_obj] = 1.0
            else:
                omega = np.random.dirichlet(np.ones(self.n_obj)).astype(np.float32)

        self.current_omega = omega
        self.episode_count += 1
        return omega

    # ── Action Selection ─────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, omega: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        omega_t = torch.FloatTensor(omega).unsqueeze(0).to(DEVICE)
        action  = self.actor(obs_t, omega_t).cpu().numpy()[0]
        if explore:
            noise = np.random.normal(0, self.expl_noise, size=action.shape)
            # Clip per range: pose dims [-1,1], BW dims [0,1]
            action[:4] = np.clip(action[:4] + noise[:4], -1.0,  1.0)
            action[4:] = np.clip(action[4:] + noise[4:],  0.0,  1.0)
            self.expl_noise = max(self.expl_noise_min,
                                  self.expl_noise * self.expl_noise_decay)
        return action

    # ── TD3 Update ───────────────────────────────────────────────

    def update(self, batch_size: int = 256) -> dict:
        if len(self.buffer) < batch_size:
            return {}

        obs_b, act_b, r_b, obs_n_b, omega_b, done_b = self.buffer.sample(batch_size)

        with torch.no_grad():
            # Target policy smoothing (TD3) — clamp per range
            noise  = (torch.randn_like(act_b) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip)
            next_a_raw = self.actor_target(obs_n_b, omega_b) + noise
            # Pose dims clamp [-1,1], BW dims clamp [0,1]
            n_pose = self.actor.n_pose_dims
            next_a = torch.cat([
                next_a_raw[..., :n_pose].clamp(-1., 1.),
                next_a_raw[..., n_pose:].clamp(0.,  1.),
            ], dim=-1)

            # Twin critic targets — take min per objective
            Q1_next = self.critic1_target(obs_n_b, next_a, omega_b)  # (B, n_obj)
            Q2_next = self.critic2_target(obs_n_b, next_a, omega_b)
            Q_next  = torch.min(Q1_next, Q2_next)                    # (B, n_obj)

            # Bellman target (vector-valued, NO scalarization here)
            target = r_b + self.gamma * (1.0 - done_b) * Q_next   # (B, n_obj)

        # Critic loss: VECTOR MSE — train all objectives every transition
        # (Scalarizing here would starve objectives not in current omega)
        Q1 = self.critic1(obs_b, act_b, omega_b)
        Q2 = self.critic2(obs_b, act_b, omega_b)
        c_loss = F.mse_loss(Q1, target) + F.mse_loss(Q2, target)

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 10.)
        self.critic_opt.step()
        self.total_updates += 1

        info = {"critic_loss": float(c_loss)}

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
    def eval_omega(self, env, omega: np.ndarray,
                   n_episodes: int = 5, steps: int = 150) -> np.ndarray:
        """Evaluate policy for one omega. Returns mean reward vector."""
        uav_ids = sorted(env.reset().keys())
        uid     = uav_ids[0]
        all_rv  = []
        for _ in range(n_episodes):
            obs_dict = env.reset()
            obs      = obs_dict[uid]
            ep_rv    = []
            done     = False
            t        = 0
            while not done and t < steps:
                action   = self.select_action(obs, omega, explore=False)
                obs_dict, _, done, info = env.step({uid: action})
                obs = obs_dict.get(uid, obs)
                rv  = info.get("reward_vector", None)
                if rv is not None:
                    ep_rv.append(np.array(rv, dtype=np.float32))
                t += 1
            if ep_rv:
                all_rv.append(np.mean(ep_rv, axis=0))
        return np.mean(all_rv, axis=0) if all_rv else np.zeros(self.n_obj, dtype=np.float32)

    @torch.no_grad()
    def gpi_select_action(self, obs: np.ndarray, omega_test: np.ndarray) -> np.ndarray:
        """
        GPI inference: evaluate actor with each (omega_i) in bank,
        pick action that maximizes omega_test^T * Q(s, a_i, omega_i).
        Falls back to standard select_action if bank is empty.
        """
        if self.bank is None or len(self.bank) == 0:
            return self.select_action(obs, omega_test, explore=False)

        obs_t      = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        omega_t    = torch.FloatTensor(omega_test).unsqueeze(0).to(DEVICE)
        best_val   = -float("inf")
        best_action = None

        # Also consider current actor with test omega
        all_entries = list(zip(self.bank.actor_states, self.bank.omegas))

        for actor_sd, omega_i in all_entries:
            # Temporarily load snapshot
            tmp = copy.deepcopy(self.actor)
            tmp.load_state_dict(actor_sd)
            tmp.eval()
            omega_i_t = torch.FloatTensor(omega_i).unsqueeze(0).to(DEVICE)
            a_i = tmp(obs_t, omega_i_t)                         # action under omega_i
            q   = self.critic1(obs_t, a_i, omega_i_t)           # Q-vector (1, n_obj)
            val = (q * omega_t).sum(-1).item()                  # scalarized by test omega
            if val > best_val:
                best_val    = val
                best_action = a_i.cpu().numpy()[0]

        # Also try current actor with test omega
        a_cur = self.actor(obs_t, omega_t)
        q_cur = self.critic1(obs_t, a_cur, omega_t)
        val_cur = (q_cur * omega_t).sum(-1).item()
        if val_cur > best_val:
            best_action = a_cur.cpu().numpy()[0]

        return best_action

    @torch.no_grad()
    def eval_omega_gpi(self, env, omega: np.ndarray,
                       n_episodes: int = 5, steps: int = 150) -> np.ndarray:
        """Evaluate GPI policy (bank + current actor) for one omega."""
        uav_ids = sorted(env.reset().keys())
        uid     = uav_ids[0]
        all_rv  = []
        for _ in range(n_episodes):
            obs_dict = env.reset()
            obs      = obs_dict[uid]
            ep_rv    = []
            done     = False
            t        = 0
            while not done and t < steps:
                action   = self.gpi_select_action(obs, omega)
                obs_dict, _, done, info = env.step({uid: action})
                obs = obs_dict.get(uid, obs)
                rv  = info.get("reward_vector", None)
                if rv is not None:
                    ep_rv.append(np.array(rv, dtype=np.float32))
                t += 1
            if ep_rv:
                all_rv.append(np.mean(ep_rv, axis=0))
        return np.mean(all_rv, axis=0) if all_rv else np.zeros(self.n_obj, dtype=np.float32)

    def eval_pareto_front(self, env, n_omega: int = 50,
                          n_episodes: int = 5, steps: int = 150,
                          use_gpi: bool = False):
        """Sweep n_omega weights, eval each. use_gpi=True uses GPI inference."""
        corners  = np.eye(self.n_obj, dtype=np.float32)
        interior = np.random.dirichlet(np.ones(self.n_obj),
                                       n_omega - self.n_obj).astype(np.float32)
        omegas   = np.vstack([corners, interior])
        results  = []
        eval_fn  = self.eval_omega_gpi if use_gpi else self.eval_omega
        for i, w in enumerate(omegas):
            rv = eval_fn(env, w, n_episodes, steps)
            results.append(rv)
            print(f"  eval {i+1}/{len(omegas)}: w={np.round(w,2)} "
                  f"QoS={rv[0]:.3f} Eng={rv[1]:.3f} Fair={rv[2]:.3f}")
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
            "expl_noise":     self.expl_noise,
            "total_updates":  self.total_updates,
            "episode_count":  self.episode_count,
            "omega_sampling": self.omega_sampling,
            "current_omega":  self.current_omega.tolist(),
            # OLS state
            "ols_omegas":  [w.tolist() for w in self.ols.omegas]    if self.ols else [],
            "ols_values":  [v.tolist() for v in self.ols.values]    if self.ols else [],
            "ols_phase":   self.ols.phase                           if self.ols else None,
            "ols_corner":  self.ols.corner_idx                      if self.ols else 0,
            # Policy bank (gpi_ols mode)
            "bank_actor_states": self.bank.actor_states if self.bank else [],
            "bank_omegas":       [w.tolist() for w in self.bank.omegas] if self.bank else [],
        }, path)
        if self.bank:
            print(f"[MOMA-AC] Saved: {path}  (bank={len(self.bank)} policies)")
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
        self.expl_noise    = st.get("expl_noise",    self.expl_noise)
        self.total_updates = st.get("total_updates", 0)
        self.episode_count = st.get("episode_count", 0)
        self.current_omega = np.array(st.get("current_omega", self.current_omega),
                                      dtype=np.float32)
        if self.ols and st.get("ols_omegas"):
            self.ols.omegas     = [np.array(w, np.float32) for w in st["ols_omegas"]]
            self.ols.values     = [np.array(v, np.float32) for v in st["ols_values"]]
            self.ols.phase      = st.get("ols_phase",  "corner")
            self.ols.corner_idx = st.get("ols_corner", 0)
        if self.bank and st.get("bank_actor_states"):
            self.bank.actor_states = st["bank_actor_states"]
            self.bank.omegas       = [np.array(w, np.float32) for w in st["bank_omegas"]]
            print(f"[MOMA-AC] Loaded bank: {len(self.bank)} policies")
        print(f"[MOMA-AC] Loaded: {path} "
              f"(ep={self.episode_count}, updates={self.total_updates})")
