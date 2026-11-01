# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MOMA-AC: a preference-conditioned, multi-objective TD3 agent that controls a UAV base station in a simulated network-slicing environment (eMBB/URLLC/mMTC). It learns a Pareto front over three objectives: QoS, energy, and fairness. The research question is whether GPI-based weight selection (GPI-LS, GPI-PD) beats the original MOMA-AC's uniform omega sampling, which is the baseline.

## Commands

There is no test suite, linter, or build step. Run everything from the repo root, because `environment.py` loads `config/environment/default.yaml` by relative path at import time.

```powershell
py -m pip install -r requirements.txt

# Train one variant
py train_moma_ac.py --config config/train/moma_ac_random.yaml   # baseline
py train_moma_ac.py --config config/train/moma_ac_gpi_ls.yaml
py train_moma_ac.py --config config/train/moma_ac_gpi_pd.yaml

# Optional flags
py train_moma_ac.py --config <yaml> --env_config <yaml> --checkpoint <path.pth> --seed <n> --resume --max_hours <h>

# Compare variants (mean ± std over seeds)
py compare_runs.py saved_models_moma_ac_random saved_models_moma_ac_gpi_ls saved_models_moma_ac_gpi_pd
```

Use `py` on this machine; `python` is not on the PATH in Git Bash.

Long runs go on Kaggle: see `kaggle/README.md`. Rebuild the upload with `py kaggle/build_bundle.py` after any code change; a new file the training needs must also be added to `FILES` there. `MOMA_DEVICE=cpu|cuda` overrides the device. `kaggle/run_kaggle.py` runs several training processes at once and pins each to the least busy GPU.

**Smoke test:** to check a change, copy a train config and set `total_training_steps: 1500`, `min_buffer_size: 100`, `batch_size: 32`, `max_steps_per_episode: 20`, `save_interval: 1500`, `eval_interval: 750`, `eval_n_omega: 5`, `eval_n_episodes: 1`, `eval_steps: 5`, and a throwaway `save_dir`. For the GPI variants also set `gpi_ls_interval: 300`, `gpi_ls_episodes: 1`, `dyna_interval: 100`, `dyna_rollouts: 100`, `model_fit_steps: 30`. This takes about a minute and exercises training, GPI-LS, Dyna, evaluation, and checkpointing. Keys left out fall back to the defaults in `MOMA_AC_Trainer.__init__`.

Each run writes to `save_dir` (or `save_dir/seed_<n>` with `--seed`): `step_metrics.csv`, `eval_metrics.csv`, `pareto_points.csv` (every evaluated omega and its reward vector), `gpi_ls_metrics.csv` (GPI variants), and `checkpoints/`. Results are tables only; nothing plots. There is no `.gitignore`, so `__pycache__/` and `saved_models_*` show up as untracked.

## Architecture

The data flow is `train_moma_ac.py` (loop, metrics) ↔ `agents_moma_ac.py` (agent) ↔ `environment.py` (simulator). From `utils.py`, only `Configuration` is used.

**Environment step (`NetworkSlicingEnv.step`).** Takes `{uav_id: action}` and runs: apply pose and power → move UEs and hotspots → associate UEs to UAVs → rebuild demand areas → apply bandwidth → allocate RBs → refresh SINR, throughput, queue, delay, and PER → compute rewards. Each step represents `T_L` = 60 simulated seconds. The default config has one UAV. `done` is true only when the battery runs out.

**Radio model.** Transmit power is split equally across the RBs in use, so SINR is per RB and noise is per RB (`noise_psd + NF + 10log10(rb_bandwidth)`). Rate is attenuated Shannon, capped at `max_spectral_efficiency`. Numerology is NR n78: 3.5 GHz, 273 RBs of 360 kHz.

**Observation: 80 dims.** 5 UAV state (position, power, battery) + 9 DAs × 7 features + 4 handover + 8 surroundings. The `_get_observations` docstring (112 dims, 10 features per DA) is out of date; the assertion in the code is correct.

**Action: 13 dims.**
- `[0:2]` are horizontal deltas, scaled by `velocity_max * T_L` (the vector is capped at 900 m per step). `[2]` is the altitude delta, scaled by `velocity_max_vertical * T_L`.
- `[3]` is power. The actor outputs `[0, 1]` and the env clips to `[0.05, 1]`.
- `[4:13]` are bandwidth fractions per DA, in `_form_demand_areas` order (slice eMBB/URLLC/mMTC × distance Near/Medium/Far). They are scaled down if the total exceeds the available RBs.

The actor uses tanh for the first 3 dims and sigmoid for the rest. `select_action` and `update` clip to these same ranges, so keep them in sync.

**Rewards.** The agent trains on `info["reward_vector"]` = `[qos, -energy, fairness]`, each in a unit range and unnormalized by the agent. The scalar reward returned by `step` and `reward_weights` in the env config do not affect training.
- Energy is the average power draw over `T_L`, min-max normalized by `_init_energy_bounds`. Propulsion follows Zeng et al. 2019, so hovering (168 W) costs more than cruising (about 128 W at the best speed). Transmit draw is `P_tx / pa_efficiency`.
- Fairness is Jain's index over all active UEs, with uncovered UEs at zero.

**Agent (`MOMA_AC_Agent`).** TD3 with omega concatenated to the inputs of both actor and critics. Critic targets and losses stay vector-valued, one Q per objective; the target uses whichever twin has the lower scalarized value. Only the actor loss scalarizes (`Q · omega`), using `critic1`.

**Omega sampling (`omega_sampling`):**
- `random`: Dirichlet(1,1,1) per episode, the original MOMA-AC and the baseline. `corner_warmup_episodes` can prepend corner weights but defaults to 0.
- `gpi_ls`: GPI-PD Alg. 2 without the model. Omega is drawn per episode from the support M (`GPILinearSupport`, starts at the extrema). Every `gpi_ls_interval` steps, `update_weight_support` evaluates M, drops weights that are best at no corner weight (extrema are kept), and adds the top-k corner weights by `v^GPI_w − max_π v^π_w`. Behaviour follows the GPI policy (`gpi_select_action`, Eq. 21) plus noise. Each update trains on the batch twice: half with the current omega, half with `w' ~ M`.
- `gpi_pd`: `gpi_ls` plus GPI-prioritized replay (`PrioritizedReplayBuffer`, Eq. 10 and 17, with a Huber critic loss) and Dyna (`DynamicsEnsemble`, every `dyna_interval` steps, one-step rollouts into `model_buffer`, mixed into batches by `dyna_beta`). The ensemble members are one batched network (`EnsembleLinear`), about 4x faster on GPU than separate nets, and each member still draws its own batch.

There are no actor snapshots: the "policies" of GPI are the one conditioned network at each weight in M. `_gpi_action_batch` and `_gpi_td_errors` score each candidate only with its own `Q(s, a, w')` instead of the full K×K sweep, for speed.

**Evaluation.** `eval_pareto_front` sweeps a fixed set of weights (corners plus seeded Dirichlet samples) on `self.eval_env`, averages reward vectors, and computes exact 3D hypervolume in `train_moma_ac.py`, plus EUM, EU over the front, and cardinality. `eval_omega` reseeds per episode with `EVAL_SEED` (common random numbers) and restores the RNG, so values are comparable across weights, checkpoints, and variants. GPI variants also log the plain actor as `*_actor` columns; compare those with the baseline to separate the effect of sampling from the effect of GPI inference.

## Known pitfalls

- **Any env change invalidates earlier results.** Reward scales move, so old HV values and any baseline run outside this repo aren't comparable. Rerun all variants.
- **`--checkpoint` is only a partial resume.** It loads weights and optimizers, but the step counter restarts at 0 and the buffers are empty. For a real continuation use `--resume`, which reads `<save_dir>/resume/` (written at `save_interval` and when `--max_hours` runs out). It restarts the episode, since env state isn't saved. A finished run deletes `resume/` and writes a `DONE` file.
- **`--env_config` is only partly honored.** The A2G and beam-angle constants used by the numba SINR functions always come from `default.yaml`, at import.
- **`num_das_per_slice` must stay 3.** The trainer derives `act_dim` from it, but the env hardcodes 3 distance levels per slice.
- **GPI-LS evaluation costs env steps that aren't counted.** Each iteration runs `|M| + corners` × `gpi_ls_episodes` × `eval_steps` steps on `eval_env`, and `step` only counts training steps.
- **Reliability is lenient to the formula's label.** `_calculate_reliability_satisfaction` uses `(1 - PER)^4`, commented as "4 retransmissions", but that is the probability four transmissions all succeed. HARQ would be `1 - PER^4`, which is nearly always 1 at these packet sizes, so the formula was left as a signal for low SINR.
- **Some files and code are stale.** `config/agents/morl_default.yaml` is not loaded by anything, and neither is `ue_dynamics.patterns`. Six of the seven per-DA observation features are constant one-hots. Comments mention `train_morl.py`, `agents_morl.py`, and `visualizer.py`, which are not in this repo. The OpenGL model classes and `QueueingModel` in `utils.py` are unused.
