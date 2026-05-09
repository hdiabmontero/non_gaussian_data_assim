# non_gaussian_data_assim

A library and experiment harness for ensemble data assimilation (DA) with non-Gaussian filters. Companion code to *"Ensemble Kalman, Adaptive Gaussian Mixture, and Particle Flow Filters for Optimized Earthquake Forecasting"* (Computers and Geosciences).

It provides:

- Three ensemble DA methods: Ensemble Kalman Filter, Adaptive Gaussian Mixture Filter, Particle Flow Filter.
- Three forward-model test cases: Lorenz 63, Lorenz 96, Kuramoto–Sivashinsky.
- A unified, [Hydra](https://hydra.cc/)-driven experiment script that lets you mix and match cases and DA methods from the command line.
- Trajectory and ensemble metrics (RMSE, MAE, MAPE, CRPS).
- A pytest suite that smoke-tests every (case, DA method) combination.

---

## Installation

The project is managed with `[uv](https://docs.astral.sh/uv/)`.

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs the project plus dev dependencies (pytest, mypy, etc.).

To run any command inside the environment, prefix with `uv run`.

---

## Quick start

Run the default experiment (Lorenz 63 + EnKF):

```bash
uv run python scripts/main.py
```

Pick a different case and DA method:

```bash
uv run python scripts/main.py case=lorenz_96 da_method=pff
uv run python scripts/main.py case=kuramoto da_method=agmf
```

Override common settings on the command line:

```bash
uv run python scripts/main.py case=kuramoto da_method=enkf \
    data_assimilation_steps=50 ensemble_size=100 seed=7
```

Override nested DA-method parameters:

```bash
uv run python scripts/main.py case=lorenz_96 da_method=enkf \
    da_method.inflation_factor=2.0 da_method.localization_distance=8
```

The script prints the resolved config, runs the DA loop, prints a metrics table, and shows plots.

---

## Configuration layout

All experiment knobs live under `[configs/](configs/)`:

```
configs/
├── config.yaml             # root: defaults list + common settings (seed, fallbacks)
├── case/
│   ├── lorenz_63.yaml      # Lorenz 63 case
│   ├── lorenz_96.yaml      # Lorenz 96 case
│   └── kuramoto.yaml       # Kuramoto–Sivashinsky case
└── da_method/
    ├── enkf.yaml           # Ensemble Kalman Filter
    ├── agmf.yaml           # Adaptive Gaussian Mixture Filter
    └── pff.yaml            # Particle Flow Filter
```

### How composition works

`configs/config.yaml` declares a Hydra `defaults` list:

```yaml
defaults:
  - _self_
  - case: lorenz_63
  - da_method: enkf
```

Selecting `case=…` swaps in one of the files in `configs/case/`; selecting `da_method=…` swaps in one of the files in `configs/da_method/`.

Each `case/*.yaml` uses `# @package _global_` so it can both:

1. Provide its own sensible defaults for the common settings (`data_assimilation_steps`, `model_integration_steps`, `ensemble_size`, `inflation_factor`, `localization_distance`), and
2. Define a fully specified `case:` block (forward model, observation operator, initial-state generator, prior-ensemble generator, plotter) used by `scripts/main.py`.

Each `case/*.yaml` also declares a `da_method_overrides` block. At runtime, `scripts/main.py` does:

```python
da_method_cfg = OmegaConf.merge(cfg.da_method, cfg.case.da_method_overrides[cfg.da_method.name])
```

so case-specific tunings (e.g. localization radius for Lorenz 96, regularization for PFF on Kuramoto) override the defaults from the `da_method/` group without duplicating the full method spec.

### Everything is built via `hydra.utils.instantiate`

Forward model, observation operator, initial state, prior ensemble, DA method, and plotter all carry `_target_` and are instantiated by Hydra. Initial-state and prior-ensemble generators use `_partial_: true` so the script can supply runtime arguments (`rng_key`, `ensemble_size`).

This means you can swap implementations purely from YAML/CLI — no Python edits required.

### Common (case-agnostic) settings

Lives at the root of `configs/config.yaml` (and in each case file as a `# @package _global_` override):


| Key                       | Meaning                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `seed`                    | PRNG seed (`jax.random.PRNGKey(seed)`).                                       |
| `data_assimilation_steps` | Number of outer DA cycles.                                                    |
| `model_integration_steps` | Number of forward-model sub-steps per DA cycle.                               |
| `ensemble_size`           | Number of ensemble members / particles.                                       |
| `inflation_factor`        | Default covariance inflation, referenced by case overrides via interpolation. |
| `localization_distance`   | Default localization radius (case-specific).                                  |


---

## Implemented cases


| Case (`case=…`) | State                              | Forward model                                                    | Default obs                        |
| --------------- | ---------------------------------- | ---------------------------------------------------------------- | ---------------------------------- |
| `lorenz_63`     | 3-D chaotic ODE (`x, y, z`)        | RK4 integrator with `σ=10, β=8/3, ρ=28`                          | All three components, `R = 5·I`    |
| `lorenz_96`     | 50-D ring of variables             | RK4 with forcing `F=8`, periodic boundary                        | Every 2nd grid point, `R = 0.25·I` |
| `kuramoto`      | 512-D PDE on a periodic 1-D domain | Pseudo-spectral exponential time differencing, domain length 100 | Every 4th grid point, `R = 0.1·I`  |


Each case ships:

- A forward-model class (`src/non_gaussian_data_assim/forward_models/…`).
- An observation operator (linear, selecting a subset of grid points).
- An initial-state generator and a prior-ensemble generator (`src/non_gaussian_data_assim/initial_conditions.py`).
- A plotter that's appropriate for the dimensionality (`plot_low_dim_trajectory` for L63, `plot_high_dim_field` for L96 / Kuramoto).
- Per-DA-method tuning overrides.

To add a case, drop a new file under `configs/case/` and (if needed) add a corresponding forward model in `src/non_gaussian_data_assim/forward_models/`.

---

## Implemented DA methods


| Method (`da_method=…`) | Class                           | Key parameters                                                                                                                          |
| ---------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `enkf`                 | `EnsembleKalmanFilter`          | `inflation_factor`, `localization_distance`                                                                                             |
| `agmf`                 | `AdaptiveGaussianMixtureFilter` | `inflation_factor`, `nc_threshold`, `localization_distance`, `w_prev`                                                                   |
| `pff`                  | `ParticleFlowFilter`            | `num_pseudo_time_steps`, `step_size`, `stepper`, `kernel_type`, `localization_distance`, `inflation_factor`, `prior_cov_regularization` |
| `particle_filter`      | `ParticleFilter`                | `resample_threshold`, `jitter_scale`                                                                                                    |


All four derive from `BaseDataAssimilationMethod` (`src/non_gaussian_data_assim/da_methods/base.py`) and follow the same forecast → analysis → update interface, so the `scripts/main.py` driver works for any of them without special-casing.

### Notes on Particle Flow Filter

PFF inverts the empirical prior covariance directly. With small ensembles in high-dimensional cases (e.g. Kuramoto with `ensemble_size=50`, `state_dim=512`) the empirical covariance is rank-deficient and even with localization its condition number is ≈ 10¹³, which makes the inverse numerically explosive.

To stabilize this, `ParticleFlowFilter` takes an optional `prior_cov_regularization` parameter:

- `None` (default) — no regularization, original behavior. Fine for L63 and L96.
- A float `r` — replaces `prior_cov` with `prior_cov + r · mean(diag(prior_cov)) · I` before inversion. Scale-aware Tikhonov regularization. The Kuramoto case sets this to `1.0e-2`.

Add it from the CLI as `da_method.prior_cov_regularization=1e-2`.

### Notes on the bootstrap Particle Filter

`particle_filter` is the standard sequential-importance-resampling (SIR) bootstrap particle filter. Each analysis step:

1. Computes Gaussian-likelihood weights `w_i ∝ exp(-½ (y - H x_i)ᵀ R⁻¹ (y - H x_i))` (log-sum-exp normalized for numerical stability).
2. Computes the effective sample size `N_eff = 1 / Σ w_i²`.
3. Performs **systematic resampling** when `N_eff < resample_threshold · ensemble_size`.
4. Optionally adds Gaussian jitter `jitter_scale · N(0, I)` to resampled particles to combat sample impoverishment after resampling.

Two knobs:

- `resample_threshold` — fraction of `ensemble_size` below which to resample. `0.5` is a common choice; set to `1.0` to resample every step.
- `jitter_scale` — magnitude of the post-resampling jitter (in the *same units as the state*). `0` disables it. Higher-dimensional cases need carefully tuned values: too small and particles collapse, too large and you destroy the posterior structure.

The bootstrap PF is asymptotically exact but suffers the curse of dimensionality: weight degeneracy gets exponentially worse with state size. For Lorenz 63 it works well; for Lorenz 96 it is mostly a baseline; for Kuramoto it only works because the ensemble is concentrated on a low-dimensional manifold (cosine-mode prior).

To add a new DA method:

1. Subclass `BaseDataAssimilationMethod` in `src/non_gaussian_data_assim/da_methods/<your_method>.py`. Implement `_analysis_step`.
2. Drop a `configs/da_method/<your_method>.yaml` with `_target_` pointing at your class plus method-specific defaults.
3. Optionally add a per-case override under `da_method_overrides.<your_method>` in any case YAML.

---

## Ensemble state shapes

The pipeline uses a consistent shape convention. `num_states` is the number of physical fields per grid (e.g. velocity-x, velocity-y, temperature) — `1` for every case shipped today. `state_dim` is the spatial dimension of one field.


| Array                 | Shape                                           | Notes                                                                                            |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Initial state (truth) | `[1, num_states, state_dim]`                    | Single member — leading 1 lets the same forward-model code handle truth and ensembles uniformly. |
| Truth trajectory      | `[1, T_total, num_states, state_dim]`           | `T_total = data_assimilation_steps · model_integration_steps + 1` when inner steps are returned. |
| Prior ensemble        | `[ensemble_size, num_states, state_dim]`        | Sampled by the case-specific generator in `initial_conditions.py`.                               |
| Reference / posterior | `[ensemble_size, T_acc, num_states, state_dim]` | Time axis grows by `model_integration_steps` per DA cycle as inner steps are concatenated.       |
| Observations          | `[data_assimilation_steps, num_obs]`            | `num_obs = len(obs_states) · len(obs_indices)`. One observation vector per outer step.           |
| `R` (obs covariance)  | `[num_obs, num_obs]`                            | Diagonal; built from `obs_noise_variance` in the case file.                                      |


The convention `[ensemble, time, num_states, state_dim]` is what the metrics expect (see below), so any new DA method or forward model should preserve it.

---

## Metrics

Implemented in `src/non_gaussian_data_assim/metrics/`. Two families share a common pattern: subclasses define `compute(...)` for the smallest unit; `__call__` `vmap`s over the missing axes and then applies the configured aggregation.

### Trajectory metrics (`metrics/trajectory_metrics.py`)

Operate on a `[ensemble, time, num_states, state_dim]` prediction against a `[time, num_states, state_dim]` truth. Each metric reduces to a scalar per (ensemble member, time) pair, then aggregates.


| Metric | Definition                                | Notes                                                                                                |
| ------ | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `RMSE` | `sqrt(mean((pred - truth)²))`             | Root mean squared error over the state.                                                              |
| `MAE`  | `mean(abs(pred - truth))`                 | Mean absolute error.                                                                                 |
| `MAPE` | `mean(abs((pred - truth) / (truth + ε)))` | Mean absolute percentage error; the small `ε` guards divides by zero. Sensitive to near-zero truths. |


All trajectory metrics accept `ensemble_aggregation` and `time_aggregation` keyword arguments. Allowed values are `"none"`, `"mean"`, `"median"`, `"max"`, `"min"`, `"std"`, `"var"`. Default `"none"` returns the per-member, per-time array; `"mean"` collapses that axis. The pipeline uses `("mean", "mean")` to print one scalar per metric.

### Ensemble metrics (`metrics/ensemble_metrics.py`)

Score the *whole ensemble* (not a single member) against a truth at each time step.


| Metric | Definition                                                        | Notes                                                                                                                                                   |
| ------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CRPS` | Continuous Ranked Probability Score, computed pointwise per state | Combines accuracy and ensemble spread into one number. Lower is better. Implemented from the empirical-CDF form, vmapped per time step then aggregated. |


CRPS supports a `time_aggregation` argument (`"none"` or any of the methods above). The pipeline uses `time_aggregation="mean"`.

### Output

`scripts/main.py` reports both **reference** (no-DA ensemble rolled out from the prior) and **posterior** (DA-corrected) metrics side by side via `print_metrics_table`, so you can immediately see whether the filter is improving over the free-running baseline.

---

## What `scripts/main.py` does

1. Seed the PRNG.
2. Instantiate the forward model and observation operator.
3. Sample the truth's initial state, roll out the truth trajectory.
4. Generate noisy observations from the truth (`observations.observation_utils.generate_observations`).
5. Compose `cfg.da_method ⊕ cfg.case.da_method_overrides[cfg.da_method.name]` and instantiate the DA method.
6. Sample a reference ensemble (no DA), roll it out for comparison.
7. Run the DA loop: forecast → analysis → record posterior; bail out with a warning if NaNs appear.
8. Compute reference and posterior metrics (RMSE, MAE, MAPE, CRPS).
9. Plot via the case-specific plotter.

All steps that build objects use `hydra.utils.instantiate`, so configuration alone determines the experiment.

---

## Testing

A pytest suite at `[tests/test_main.py](tests/test_main.py)` parametrizes over every `(case, da_method)` combination — 9 in total — and runs `scripts/main.py` end-to-end as a subprocess with small problem sizes. A run is considered passing if the script exits cleanly (returncode `0`).

Default test parameters (chosen for speed, not assimilation quality):

```text
data_assimilation_steps = 10
model_integration_steps = 5
ensemble_size           = 50
```

Run the suite:

```bash
uv run pytest tests/ -v
```

Run a single combination:

```bash
uv run pytest tests/test_main.py -v -k "kuramoto and pff"
```

The tests don't validate metric values; they catch regressions in:

- Hydra config composition (missing keys, wrong interpolation).
- `instantiate` call signatures (wrong / missing constructor arguments after refactors).
- Numerical blowups (NaN posterior triggers a non-zero exit because of the downstream metric vmap shape mismatch).
- Plotting wiring (the non-interactive `MPLBACKEND=Agg` exercises the plotters without opening windows).

If you add a new case or DA method, extend `CASES` / `DA_METHODS` at the top of `tests/test_main.py`.

---

## Project structure

```
.
├── configs/                          # Hydra config tree (see above)
├── scripts/
│   ├── main.py                       # unified Hydra-driven entrypoint
│   ├── da_lorenz_63.py               # legacy standalone script
│   ├── da_lorenz_96.py               # legacy standalone script
│   ├── da_kuramoto.py                # legacy standalone script
│   └── archive/                      # older, unmaintained scripts
├── src/non_gaussian_data_assim/
│   ├── da_methods/                   # EnKF, AGMF, PFF, base class
│   ├── forward_models/               # Lorenz 63/96, Kuramoto–Sivashinsky, …
│   ├── observations/
│   │   ├── observation_operator.py   # Linear / Nonlinear obs operators
│   │   └── observation_utils.py      # generate_observations()
│   ├── metrics/                      # trajectory + ensemble metrics
│   ├── initial_conditions.py         # initial-state / prior-ensemble generators
│   ├── plotting.py                   # plot_low_dim_trajectory, plot_high_dim_field
│   ├── localization.py               # distance-based covariance localization
│   ├── kernels.py                    # kernels for PFF
│   ├── time_integrators.py           # RK4, forward Euler, rollout helpers
│   └── jax_utils.py
├── tests/
│   └── test_main.py                  # parametrized smoke tests
├── pyproject.toml
└── uv.lock
```

The legacy `scripts/da_*.py` files predate the unified `main.py` and are kept for reference; new work should go through `scripts/main.py` and the `configs/` tree.

---

## Common usage patterns

### Sweep DA methods on one case

```bash
for da in enkf agmf pff; do
    uv run python scripts/main.py case=kuramoto da_method=$da
done
```

### Sweep over a parameter

Hydra supports its own multirun via `--multirun` / `-m`:

```bash
uv run python scripts/main.py -m \
    case=lorenz_96 da_method=enkf \
    da_method.inflation_factor=1.0,1.5,2.0,2.5
```

### Running headless (no plot windows)

Set `MPLBACKEND=Agg`. This is what the test suite does.

### Reproducibility

Pass an explicit seed, e.g. `seed=12345`. The same seed reproduces the truth, observations, and ensembles deterministically.

---

## Contributing

We use a standard fork-or-feature-branch workflow on top of `git`. The steps below assume you already have `[git](https://git-scm.com/)` and the [GitHub CLI (`gh`)](https://cli.github.com/) installed.

### 1. Clone the repository

```bash
git clone https://github.com/<owner>/non_gaussian_data_assim.git
cd non_gaussian_data_assim
uv sync
```

### 2. Create a new branch

Always work on a topic branch off `main`. Use a short, descriptive name.

```bash
git checkout -b my-feature
```

### 3. Make your changes

Edit code, add tests under `tests/`, run the suite locally:

```bash
uv run pytest tests/ -v
```

### Pre-commit hooks

The repository ships a [`pre-commit`](https://pre-commit.com/) configuration ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) that runs automatically on every `git commit`. The hooks installed are:

- **Built-ins** — `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `check-merge-conflict`.
- **`black`** — Python formatting (line length 88, Python 3.13).
- **`isort`** — import sorting (`black` profile).
- **`mypy`** — static type checking with `--ignore-missing-imports`. mypy strictness is configured in `[tool.mypy]` in [`pyproject.toml`](pyproject.toml).

Both `archive/` and `scripts/archive/` are excluded from every hook.

Install the git hook the first time you clone:

```bash
uv run pre-commit install
```

Run all hooks against every tracked file (e.g. before opening a PR):

```bash
uv run pre-commit run --all-files
```

Run a single hook:

```bash
uv run pre-commit run mypy --all-files
uv run pre-commit run black --all-files
```

If a hook fails, fix the underlying issue and re-stage. Don't bypass with `git commit --no-verify` unless you have a very specific reason — CI also runs the hooks.

### 4. Stage and commit

Stage specific files (avoid `git add .` to keep accidental files out of commits):

```bash
git status
git add path/to/changed_file.py path/to/another_file.yaml
git commit -m "Short, imperative subject line

Optional longer body explaining the why."
```

Make small, focused commits. Each commit message should describe *why* the change exists, not just *what* changed.

### 5. Push to the remote

The first push needs `-u` to set the upstream branch:

```bash
git push -u origin my-feature
```

Subsequent pushes on the same branch are just `git push`.

### 6. Open a pull request

Open the PR against `main` with the GitHub CLI:

```bash
gh pr create --base main --fill
```

Or open it via the web UI from the branch page on GitHub. In the description, summarize the change, link any related issues, and include test results.

### 7. Iterate

Address review feedback by pushing more commits to the same branch. Once approved and CI is green, the PR can be merged.

### Implementing a new forward model

Adding a new dynamical system to the harness only requires **subclassing `BaseForwardModel`** ([`src/non_gaussian_data_assim/forward_models/base.py`](src/non_gaussian_data_assim/forward_models/base.py)) **and implementing the `one_step` method**. Everything else — single-call rollout with optional inner-step trajectories, ensemble vmapping, JIT compilation — is provided by the base class.

**`one_step` acts on a single state with shape `[num_states, state_dim]`.** You write the dynamics as if there were one trajectory; `jax.vmap` (applied inside the base class) lifts it to operate over the ensemble axis, and the base class rollout machinery composes `one_step` `model_integration_steps` times per inner call and `data_assimilation_steps` times across outer cycles. You never need to write a vmapped, batched, or rollout-aware version yourself.

```python
# src/non_gaussian_data_assim/forward_models/my_model.py
import jax.numpy as jnp

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class MyModel(BaseForwardModel):
    def __init__(self, dt: float, model_integration_steps: int, state_dim: int, ...) -> None:
        super().__init__(dt, model_integration_steps, state_dim)
        # Store any model-specific parameters here.

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Advance a single state ``x`` of shape [num_states, state_dim] by ``self.dt``."""
        ...
        return x_next  # same shape: [num_states, state_dim]
```

That's it — the base class handles `__call__` (vmapped, JIT'd inner rollout for `model_integration_steps`) and `rollout` (outer loop over `data_assimilation_steps` with optional inner-step return).

To use the new model, point a case file at it via `_target_`:

```yaml
# configs/case/my_case.yaml — under case.forward_model
forward_model:
  _target_: non_gaussian_data_assim.forward_models.my_model.MyModel
  dt: 0.01
  model_integration_steps: ${model_integration_steps}
  state_dim: 64
  # any extra constructor arguments
```

No edits to `scripts/main.py` are needed.

---

## License

MIT License Version 1, 7 April 2024.
