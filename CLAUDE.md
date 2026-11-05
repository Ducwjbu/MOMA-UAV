# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MOMA-AC: a preference-conditioned, multi-objective TD3 agent that controls a UAV base station in a simulated network-slicing environment (eMBB/URLLC/mMTC). It learns a Pareto front over three objectives: QoS, energy, and fairness. The research question is whether smarter weight selection (OLS, GPI-LS, GPI-PD) beats the original MOMA-AC's uniform omega sampling, which is the baseline. Those comparisons are on one UAV. `ma3` is the paper's multi-agent MOMA-TD3 on three UAVs, with uniform sampling only so far.

## Commands

There is no test suite, linter, or build step. Run everything from the repo root, because `environment.py` loads `config/environment/default.yaml` by relative path at import time.

```powershell
py -m pip install -r requirements.txt

# Train one variant
py train_moma_ac.py --config config/train/moma_ac_random.yaml   # baseline
py train_moma_ac.py --config config/train/moma_ac_ols.yaml
py train_moma_ac.py --config config/train/moma_ac_gpi_ls.yaml
py train_moma_ac.py --config config/train/moma_ac_gpi_pd.yaml
py train_moma_ac.py --config config/train/moma_ac_ma3.yaml      # 3 UAVs, CTDE

# No-training reference for ma3: 3 UAVs hovering in a fixed triangle (20 steps, as the ma3 eval)
py eval_formation.py

# Optional flags
py train_moma_ac.py --config <yaml> --env_config <yaml> --checkpoint <path.pth> --seed <n> --resume --max_hours <h>

# Final numbers: eval one checkpoint on 20 held-out episodes (seeds 100000+), CPU, with bootstrap CIs
py eval_checkpoint.py --config config/train/moma_ac_random.yaml --checkpoint <dir>/checkpoints/checkpoint_step_1000000.pth --out eval_random
py eval_checkpoint.py --config config/train/moma_ac_ols.yaml --checkpoint <...> --baseline eval_random_actor.npy --out eval_ols
py eval_checkpoint.py --formation --out eval_triangle

# Compare variants (mean ± std over seeds)
py compare_runs.py saved_models_moma_ac_random saved_models_moma_ac_ols saved_models_moma_ac_gpi_ls saved_models_moma_ac_gpi_pd
```

Use `py` on this machine; `python` is not on the PATH in Git Bash.

Long runs go on Kaggle: see `kaggle/README.md`. Rebuild the upload with `py kaggle/build_bundle.py` after any code change; a new file the training needs must also be added to `FILES` there. `MOMA_DEVICE=cpu|cuda` overrides the device. `kaggle/run_kaggle.py` runs several training processes at once and pins each to the least busy GPU. `ma3` is not in its default variants: pass `--variants ma3`.

**Smoke test:** to check a change, copy a train config and set `total_training_steps: 1500`, `min_buffer_size: 100`, `batch_size: 32`, `max_steps_per_episode: 20`, `save_interval: 1500`, `eval_interval: 750`, `eval_n_omega: 5`, `eval_n_episodes: 1`, `eval_steps: 5`, and a throwaway `save_dir`. For the OLS and GPI variants also set `gpi_ls_interval: 300`, `gpi_ls_episodes: 1`, `dyna_interval: 100`, `dyna_rollouts: 100`, `model_fit_steps: 30`. This takes about a minute and exercises training, GPI-LS, Dyna, evaluation, and checkpointing. Keys left out fall back to the defaults in `MOMA_AC_Trainer.__init__`. A train config can name its environment with `env_config`; `--env_config` overrides it.

Each run writes to `save_dir` (or `save_dir/seed_<n>` with `--seed`): `step_metrics.csv`, `eval_metrics.csv`, `pareto_points.csv` (every evaluated omega and its reward vector), `gpi_ls_metrics.csv` (OLS and GPI variants), and `checkpoints/`. Results are tables only; nothing plots. `.gitignore` excludes `__pycache__/`, root-level `saved_models_*/` and `runs/`, checkpoints (`*.pth`), and replay buffers (`*.npz`, `resume/`); the CSVs copied into `results/` are tracked.

## Architecture

The data flow is `train_moma_ac.py` (loop, metrics) ↔ `agents_moma_ac.py` (agent) ↔ `environment.py` (simulator). From `utils.py`, only `Configuration` is used.

**Environment step (`NetworkSlicingEnv.step`).** Takes `{uav_id: action}` and runs: apply pose and power → move UEs and hotspots → associate UEs to UAVs → rebuild demand areas → apply bandwidth → allocate RBs → refresh SINR, throughput, queue, delay, and PER → compute rewards. Each step represents `T_L` = 60 simulated seconds. The default config has one UAV. `done` is true as soon as any UAV's battery runs out. `info` also reports `coverage`, `handovers` for that step, `min_uav_distance` (inf with one UAV), and `backhaul_limited` (share of UAVs over their backhaul capacity). The handover-rate observation counts handovers in the last `T_L`, so it falls to 0 once they stop.

**Radio model.** Transmit power is split equally across the RBs in use, so SINR is per RB and noise is per RB (`noise_psd + NF + 10log10(rb_bandwidth)`). Rate is attenuated Shannon, capped at `max_spectral_efficiency`. Numerology is NR n78: 3.5 GHz, 273 RBs of 360 kHz.

**Several UAVs (`config/environment/ma3.yaml`: 3 UAVs, 150 UEs).** All UAVs reuse the same band. Interference from UAV j is its expected power on a victim's RB, `P_j / total_rbs` (zero if j uses no RBs). Without an `uav_params.antenna` section it is reduced by `sidelobe_attenuation_db` at UEs outside j's beam cone (without that key they get none), and a UE joins the nearest UAV whose cone contains it. `min_uav_separation` pushes UAVs apart after all moves, inside the fly ranges and with no reward penalty, and propulsion energy uses the final positions. A push can exceed the speed limit: a search found pushes up to about 56 m per step (a coincident pair overshoots by 50 m, 950 m against 900 m), while the team's mean speed is unaffected. A handover interrupts `D` = 50 ms of the 60 s step, so it adds the mean delay `D^2 / (2 T_L)` and scales the 1% failure loss by `D / T_L`, instead of charging every packet of the step, which makes a handover nearly free in the reward. The reward vector stays one shared team reward. None of this changes results with one UAV.

**Realistic link model (both env configs; `default.yaml` is `ma3.yaml` with 1 UAV, 50 UEs and no separation).** Parameters and sources are in the comments of the env configs. Each part is off when its key is absent, which restores the older model.
- `city` (v1a): `city.py` builds an explicit ITU-R P.1410 grid (pitch 57.7 m, footprints 40.8 m, Rayleigh heights) from the seeded global RNG at every `reset`. `_update_channel_state` runs once per step after the UE moves: LoS of every UE-UAV link is geometric (`City.line_of_sight`, exact segment-box test; an indoor UE's own building is skipped), UEs inside a footprint are indoor with the O2I loss of TR 38.901 Table 7.4.3-3, and shadowing (`channel.shadowing` sigma_los/nlos/o2i) is an AR(1) over the distance the UE plus the UAV moved, with Table 7.5-6 decorrelation distances. State lives on the UE object (`los`, `indoor`, `shadow_db`, `o2i_db`, `k_factor_db`). The backhaul's LoS is geometric too. UAVs are kept `roof_clearance` above the roof under them (`_apply_height_bounds`, after the separation push). Geometric LoS is far below Al-Hourani's S-curve for street-level UEs (street canyons), so coverage is much lower than with the statistical model.
- `channel.fast_fading`: the rate is the mean of the attenuated Shannon rate over Rician (outdoor LoS, K per UE-UAV link) or Rayleigh fading, from a table built at init with a fixed seed (`_build_fading_table`). Only with the city.
- `wind`: Dai et al. (arXiv:2304.06909). One draw per episode (`_draw_wind`), power law in height. The configs use Hanoi: Weibull from Global Wind Atlas (district mean), direction from the Lang station wind rose (`direction_frequencies`, 16 sectors; without the key, von Mises). Both configs set `rotor_speed: air`. The commanded ground velocity is cut so the airspeed stays within `velocity_max` (`_limit_airspeed`); the UAV stays below the height where the wind reaches `max_speed`, and an episode too windy even at the lowest height is redrawn (`wind_rejections`). Propulsion uses Dai's Eq. 8 at zero acceleration; `rotor_speed: ground` follows the paper (blade and induced power at ground speed), `air` uses the airspeed. Descent is limited by `velocity_max_descent`.
- `uav_params.antenna`: the 3GPP TR 38.901 element pattern pointing down, `8 - min(12 (theta/65)^2, 30)` dBi, on serving and interfering links. A UE joins the UAV with the highest reference SNR (`P / total_rbs` per RB at the UAV's current power), and is uncovered below `min_association_snr_db` (-6 dB, assumed). SINR then comes from `_sinr_db` in numpy, not the numba function.
- `channel.shadowing`: per step and UE-UAV link, the LoS state is drawn from Al-Hourani's `P_LoS`, eta takes that state's value, and log-normal shadowing of TR 36.777 UMa-AV is added (`sigma_LoS = 4.64 exp(-0.0066 h)`, `sigma_NLoS = 6` dB). Association and SINR share the step's draws (`channel_draws`). Without it eta is the expectation over the LoS state, as before.
- `a2g` is read from the env config by `_link_gains_db` and `_backhaul_capacity`. `ma3` uses Al-Hourani's dense urban values.
- `energy_models.radio_static_power` switches the radio draw to EARTH's micro BS model, `112 + 2.6 P_tx` W, instead of `P_tx / pa_efficiency`.
- Propulsion uses Table I of the arXiv version of Zeng et al. (W = 100 N, hover 1.37 kW), with a 548 Wh battery (two DJI TB60), 17 m/s, 6.3 W, NF 9 dB, and heights 50-120 m (civil limit). The battery lasts about 22 min, so `moma_ac_ma3.yaml` runs 20-step episodes and evals. The energy-optimal speed (about 21 m/s) is above the speed limit.

**Backhaul (the `backhaul` config section; absent means unlimited).** Each UAV has its own out-of-band link to one ground gateway. Without `model`, it uses the access A2G and rate models, so capacity falls with distance. With `model: mmwave` (both configs), it follows Galkin et al. (arXiv:1710.10807): path loss `c d^-alpha` by LoS state, their Eq. 3 for `P_LoS` between two elevated ends, 64.2 dB of antenna gain, and the capacity is the mean over the LoS state. It almost never binds (`backhaul_limited` is 0 for the fixed triangle), unlike the earlier assumed 20 MHz link. Load is `sum(min(access rate, offered load)) / capacity`. Above load 1 all the UAV's UEs slow down by the factor `1 / load`, which is continuous at load 1 and keeps the shares within a UAV. The queue's service rate is `min(total UE rate, capacity)` at every load, so delay and drops rise as load approaches 1. Individual UE rates can still exceed the capacity, because the queue carries the penalty rather than throughput satisfaction. With a backhaul the queue also differs in three ways, none of which apply without one: packet size is the mean over packets with no 1000-bit overhead, so `rho` is offered bits over service bits; delay is capped at the 99 ms overload value; and the finite-buffer drop formula also runs above `rho = 1`, so drops are continuous there. The gateway is at the area's edge, 30 m high. Backhaul transmit power isn't in the energy objective. With the earlier 20 MHz link, fixed triangles still did best centered in the area, so don't assume a backhaul pulls UAVs toward the gateway. With one UAV, `info` and the observation have it but `eval_metrics.csv` has no `backhaul_limited`.

**Observation.** One UAV, 80 dims: 5 UAV state (position, power, battery) + 9 DAs × 7 features + 4 handover + 8 surroundings. N > 1 UAVs, `30 + 4(N-1)` dims (38 for 3): the constant one-hots go, leaving 9 DAs × 1 feature, and uncovered UEs per direction (4) plus each other UAV's dx, dy, dh, and power ratio (4 each) are added. A `backhaul` section adds 1 dim, the UAV's load capped at 2 and halved, and a `wind` section adds 2 (the wind vector at the UAV's height over `max_speed`), so the configs have 83 (one UAV) and 41 (`ma3`).

**Action: 13 dims per UAV.**
- `[0:2]` are horizontal deltas, scaled by `velocity_max * T_L` (the vector is capped at 900 m per step). `[2]` is the altitude delta, scaled by `velocity_max_vertical * T_L`.
- `[3]` is power. The actor outputs `[0, 1]` and the env clips to `[0.05, 1]`.
- `[4:13]` are bandwidth fractions per DA, in `_form_demand_areas` order (slice eMBB/URLLC/mMTC × distance Near/Medium/Far). They are scaled down if the total exceeds the available RBs.

The actor uses tanh for the first 3 dims and sigmoid for the rest. `select_action` and `update` clip to these same ranges, so keep them in sync.

**Rewards.** The agent trains on `info["reward_vector"]` = `[qos, -energy, fairness]`, each in a unit range and unnormalized by the agent. The scalar reward returned by `step` and `reward_weights` in the env config do not affect training.
- Energy is the average power draw over `T_L`, min-max normalized by `_init_energy_bounds`. Propulsion follows Zeng et al. 2019, so hovering costs more than cruising (about 1.37 kW against 0.94 kW at the best speed, about 21 m/s, which is above the 17 m/s limit). Transmit draw is `P_tx / pa_efficiency`, or EARTH's model with `radio_static_power`.
- Fairness is Jain's index over all active UEs, with uncovered UEs at zero. Both configs set `fairness_metric: fraction_of_demand`, Jain, Chiu & Hawe 1984, Sec. 5 item 6: `x_i = min(throughput / offered load, 1)`. Without the key `x_i` is throughput over the slice's `min_rate`, uncapped; a few mMTC UEs given many RBs (up to 800x their need) then dominate it and the index falls to about 0.1.

**Agent (`MOMA_AC_Agent`).** TD3 with omega concatenated to the inputs of both actor and critics. Critic targets and losses stay vector-valued, one Q per objective; the target uses whichever twin has the lower scalarized value. Only the actor loss scalarizes (`Q · omega`), using `critic1`.

**Omega sampling (`omega_sampling`):**
- `random`: Dirichlet(1,1,1) per episode, the original MOMA-AC and the baseline. `corner_warmup_episodes` can prepend corner weights but defaults to 0.
- `ols`: same support loop and two-part update as `gpi_ls`, sharing its `gpi_*` config keys, but behaviour and evaluation use the plain actor and corner weights rank by the OLS gap (SFOLS Alg. 2): `optimistic_value` (an LP over every weight explored so far, `support.explored`, including dropped ones) minus the best measured value. It borrows only the OLS ranking: unlike SFOLS there is no separate policy per weight, no priority queue, and GPI is not used in training. It exists to test the sampling without critic estimates choosing actions.
- `gpi_ls`: GPI-PD Alg. 2 without the model. Omega is drawn per episode from the support M (`GPILinearSupport`, starts at the extrema). Every `gpi_ls_interval` steps, `update_weight_support` evaluates M, drops weights that are best at no corner weight (extrema are kept), and adds the top-k corner weights by `v^GPI_w − max_π v^π_w`. Behaviour follows the GPI policy (`gpi_select_action`, Eq. 21) plus noise. Each update trains on the batch twice: half with the current omega, half with `w' ~ M`.
- `gpi_pd`: `gpi_ls` plus GPI-prioritized replay (`PrioritizedReplayBuffer`, Eq. 10 and 17, with a Huber critic loss) and Dyna (`DynamicsEnsemble`, every `dyna_interval` steps, one-step rollouts into `model_buffer`, mixed into batches by `dyna_beta`). The ensemble members are one batched network (`EnsembleLinear`), about 4x faster on GPU than separate nets, and each member still draws its own batch.

**Several UAVs (`MOMA_CTDE_Agent`, chosen by the trainer when the env has more than one UAV).** MOMA-TD3, Alg. 1 of the MOMA-AC paper. `MultiHeadActor` has a shared trunk and one head per UAV, and `pi_i` sees only `o_i` and omega. Critics take the joint observation and joint action, flattened in UAV id order, and so does the one `buffer` (equivalent to the paper's per-agent buffers sampled in sync, since the reward is shared). The actor loss follows Eq. 10-12: for each UAV only `a_i` comes from the actor, the others keep their replay actions, and the per-UAV losses are summed. Only `omega_sampling: random` is supported; anything else raises `ValueError`. `expl_noise` decays as in the one-UAV runs, while the paper's Alg. 1 shows a fixed sigma; set `expl_noise_decay: 1.0` for that.

There are no actor snapshots: the "policies" of GPI are the one conditioned network at each weight in M. `_gpi_action_batch` and `_gpi_td_errors` score each candidate only with its own `Q(s, a, w')` instead of the full K×K sweep, for speed.

**Evaluation.** `eval_pareto_front` sweeps a fixed set of weights (corners plus seeded Dirichlet samples) on `self.eval_env`, averages reward vectors, and computes exact 3D hypervolume in `train_moma_ac.py`, plus EUM, EU over the front, and cardinality. `eval_omega` reseeds per episode with `EVAL_SEED` (common random numbers) and restores the RNG, so values are comparable across weights, checkpoints, and variants. GPI variants also log the plain actor as `*_actor` columns; compare those with the baseline to separate the effect of sampling from the effect of GPI inference. `ols` has only the plain actor, so its `hv` is already that kind of value. With several UAVs `eval_metrics.csv` and `step_metrics.csv` add `coverage`, `handovers`, `backhaul_limited`, `min_uav_distance` (the true minimum over the steps), and `mean_uav_distance` (the mean over steps of each step's closest pair). In `eval_metrics.csv` they span the sweep, with `coverage`, `handovers`, and `backhaul_limited` as a mean of episode means like the reward vector; in `step_metrics.csv` they cover only the last finished episode. Compare `ma3` with the single row `eval_formation.py` writes, which uses the same eval seeds.

The training eval (3 episodes, the same seeds every time) is for curves only. Report results from `eval_checkpoint.py`. The closed loop is chaotic: an action perturbation of 1e-6 moves a corner's QoS by about ±0.03, so the same checkpoint gives a different HV on Kaggle's GPU, a local GPU, and CPU (0.228, 0.224, and 0.222 for ma3 at 1M), and three episodes give an HV CI about 0.06 wide. Its CIs bootstrap over eval episodes, so they don't cover variance between training seeds.

## Known pitfalls

- **Any env change invalidates earlier results.** Reward scales move, so old HV values and any baseline run outside this repo aren't comparable. Rerun all variants. All results in `results/` (and the 0.228 HV quoted below) come from the earlier env: suburban channel, cone antenna, 20 N UAV, uncapped fairness, 60-step episodes, and 50 UEs for `ma3`. They were runs to check that training works.
- **`--checkpoint` is only a partial resume.** It loads weights and optimizers, but the step counter restarts at 0 and the buffers are empty. For a real continuation use `--resume`, which reads `<save_dir>/resume/` (written at `save_interval` and when `--max_hours` runs out). It restarts the episode, since env state isn't saved. A finished run deletes `resume/` and writes a `DONE` file.
- **`--env_config` is only partly honored on the numba path.** Without an `antenna` section, SINR comes from the numba function, whose A2G and beam-angle constants always come from `default.yaml`, at import. With one (`ma3`), `_link_gains_db` reads the env config. `_backhaul_capacity` reads the env config either way.
- **`num_das_per_slice` must stay 3.** The trainer derives `act_dim` from it, but the env hardcodes 3 distance levels per slice.
- **GPI-LS evaluation costs env steps that aren't counted.** Each iteration runs `|M| + corners` × `gpi_ls_episodes` × `eval_steps` steps on `eval_env`, and `step` only counts training steps.
- **`calculate_sinr_batch_numba` (the unused free-space version) still has the old interference model.** Only `calculate_sinr_batch_numba_a2g` is called.
- **`_separate_uavs` has limits.** It can give up silently after 100 passes, and a vertically stacked pair pushes only vertically. Neither is reachable with `ma3.yaml`, but they matter if `num_uavs` or the separation grows a lot.
- **A UAV missing from a partial action dict can be pushed for free.** It pays no energy for the push. The trainer always sends all UAVs.
- **The ping-pong observation dim stays 0.** `ping_pong_threshold` is 5 s and `T_L` is 60 s.
- **Execution is decentralized at network level only.** `o_i` includes neighbour positions and power and global UE direction counts, so deployment assumes those links.
- **The interference model assumes random RB use,** while the allocator fills RBs from index 0 on every UAV.
- **A UAV using 0 RBs causes no interference but still pays full transmit energy.**
- **The formation baseline starts in place with no flight cost,** while a trained team starts from random positions.
- **Two details of `MOMA_CTDE_Agent.update` come from TD3, not the paper's equations as written:** done masking in the Bellman target (Eq. 6) and the clamp after adding noise (Eq. 1-2).
- **Reliability is lenient to the formula's label.** `_calculate_reliability_satisfaction` uses `(1 - PER)^4`, commented as "4 retransmissions", but that is the probability four transmissions all succeed. HARQ would be `1 - PER^4`, which is nearly always 1 at these packet sizes, so the formula was left as a signal for low SINR.
- **Some files and code are stale.** `config/agents/morl_default.yaml` is not loaded by anything, and neither is `ue_dynamics.patterns`. Six of the seven per-DA observation features are constant one-hots. Comments mention `train_morl.py`, `agents_morl.py`, and `visualizer.py`, which are not in this repo. The OpenGL model classes and `QueueingModel` in `utils.py` are unused.
