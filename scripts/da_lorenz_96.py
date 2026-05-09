import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.base import da_rollout
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.lorenz_63 import Lorenz63Model
from non_gaussian_data_assim.forward_models.lorenz_96 import Lorenz96Model
from non_gaussian_data_assim.metrics.ensemble_metrics import CRPS
from non_gaussian_data_assim.metrics.trajectory_metrics import (
    MAE,
    MAPE,
    RMSE,
    print_metrics_table,
)
from non_gaussian_data_assim.observation_operator import LinearObservationOperator

SEED = 42

DATA_ASSIMILATION_STEPS = 50
MODEL_INTEGRATION_STEPS = 1
ENSEMBLE_SIZE = 100

DA_METHOD = "pff"
DA_METHODS = {
    "enkf": EnsembleKalmanFilter,
    "agmf": AdaptiveGaussianMixtureFilter,
    "pff": ParticleFlowFilter,
}
SPECIFIC_DA_ARGS = {
    "enkf": {
        "inflation_factor": 2.5,
        "localization_distance": 10,
    },
    "agmf": {
        "inflation_factor": 1.0,
        "localization_distance": 10,
        "nc_threshold": 0.5,
        "w_prev": np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    },
    "pff": {
        "num_pseudo_time_steps": 200,
        "step_size": 0.1,
        "stepper": "runge_kutta_4",
        "localization_distance": 5,
    },
}

# Constants and parameters
DT = 0.01
F = 8.0
T0 = 0.0

NUM_STATES = 1
STATE_DIM = 50
NUM_SKIP_OBS = 2

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM, NUM_SKIP_OBS)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 0.25


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Initial state - shape: [1, num_states, state_dim] for single ensemble member
    rng_key, key = jax.random.split(rng_key)
    X_0 = jax.random.normal(key, (1, NUM_STATES, STATE_DIM)) * 10
    X_0 = X_0.at[0, 0, -1].set(X_0[0, 0, 0])  # Periodic boundary condition

    # Define the forward model
    forward_model = Lorenz96Model(
        forcing_term=F,
        state_dim=STATE_DIM,
        dt=DT,
        model_integration_steps=MODEL_INTEGRATION_STEPS,
    )

    # Rollout the true solution
    true_sol = forward_model.rollout(
        X_0, DATA_ASSIMILATION_STEPS, return_model_integration_steps=True
    )

    # Define the observation operator
    obs_operator = LinearObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    # Generate observations
    observations = jnp.zeros((DATA_ASSIMILATION_STEPS, len(OBS_IDS)))
    for i in range(0, DATA_ASSIMILATION_STEPS):
        obs_at_t = obs_operator(
            true_sol[:, 1 + MODEL_INTEGRATION_STEPS * (i + 1)]
        )  # [1, num_obs]

        rng_key, key = jax.random.split(rng_key)
        obs_at_t = obs_at_t + jax.random.multivariate_normal(
            key, jnp.zeros(len(OBS_IDS)), R
        )  # np.sqrt(R)
        observations = observations.at[i].set(obs_at_t.flatten())  # [num_obs]

    da_model = DA_METHODS[DA_METHOD](
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        **SPECIFIC_DA_ARGS[DA_METHOD],
    )

    # Initialize the prior ensemble
    rng_key, key = jax.random.split(rng_key)
    prior_ensemble = jax.random.normal(key, (ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)) * 10
    prior_ensemble = prior_ensemble.at[:, :, -1].set(
        prior_ensemble[:, :, 0]
    )  # Periodic boundary condition

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    # Rollout the prior ensemble
    prior_ensemble = forward_model.rollout(
        prior_ensemble, DATA_ASSIMILATION_STEPS, return_model_integration_steps=True
    )

    # Perform the data assimilation
    # rng_key, key = jax.random.split(rng_key)
    # t0 = time.time()
    # posterior_ensemble = da_model.rollout(
    #     posterior_ensemble[:, 0], observations[1:], rng_key
    # )
    # t1 = time.time()
    # print(f"Time taken: {t1 - t0} seconds")

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    for i in tqdm(range(0, DATA_ASSIMILATION_STEPS)):
        rng_key, key = jax.random.split(rng_key)
        posterior_next = da_model(
            prior_ensemble=posterior_ensemble[:, -1],
            obs_vect=observations[i],
            rng_key=key,
            return_model_integration_steps=True,
        )
        if jnp.isnan(posterior_next).any():
            print(f"NaN in posterior_next at time {i}")
            break

        posterior_ensemble = jnp.concatenate(
            [posterior_ensemble, posterior_next], axis=1
        )

    # Calculate the prior and posterior errors

    rmse = RMSE(ensemble_aggregation="mean", time_aggregation="mean")
    mae = MAE(ensemble_aggregation="mean", time_aggregation="mean")
    mape = MAPE(ensemble_aggregation="mean", time_aggregation="mean")
    crps = CRPS(time_aggregation="mean")

    # Calculate the prior and posterior errors
    prior_error = {
        "rmse": rmse(prior_ensemble, true_sol[0]),
        "mae": mae(prior_ensemble, true_sol[0]),
        "mape": mape(prior_ensemble, true_sol[0]),
        "crps": crps(prior_ensemble, true_sol[0]),
    }
    posterior_error = {
        "rmse": rmse(posterior_ensemble, true_sol[0]),
        "mae": mae(posterior_ensemble, true_sol[0]),
        "mape": mape(posterior_ensemble, true_sol[0]),
        "crps": crps(posterior_ensemble, true_sol[0]),
    }

    print_metrics_table(prior_error, posterior_error, title="Lorenz 96 Metrics")

    true_sol = true_sol.reshape(
        DATA_ASSIMILATION_STEPS * MODEL_INTEGRATION_STEPS + 1, STATE_DIM
    )
    mean_prior = prior_ensemble.mean(axis=(0, 2))
    mean_post = posterior_ensemble.mean(axis=(0, 2))
    std_post = posterior_ensemble.std(axis=(0, 2))
    time_axis = np.arange(posterior_ensemble.shape[1])

    ids_to_plot = [STATE_DIM // 4, STATE_DIM // 2, 3 * STATE_DIM // 4]

    states_to_plot = zip(
        [
            true_sol,
            prior_ensemble.mean(axis=(0, 2)),
            posterior_ensemble.mean(axis=(0, 2)),
            true_sol - prior_ensemble.mean(axis=(0, 2)),
            true_sol - posterior_ensemble.mean(axis=(0, 2)),
            posterior_ensemble.var(axis=(0, 2)),
        ],
        [
            "True Solution",
            "Prior Ensemble Mean",
            "Posterior Ensemble Mean",
            "|True - Prior| difference",
            "|True - Posterior| difference",
            "Posterior Ensemble Variance",
        ],
    )

    plt.figure()
    plt.suptitle(
        f"Lorenz 96, DA Method: {DA_METHOD}, Ensemble Size: {ENSEMBLE_SIZE}, \n Prior RMSE: {prior_error['rmse']:.4f}, Posterior RMSE: {posterior_error['rmse']:.4f}"
    )

    for i, (state, state_name) in enumerate(states_to_plot):
        vmin = true_sol.min() if i < 3 else np.percentile(state, 5)
        vmax = true_sol.max() if i < 3 else np.percentile(state, 95)
        plt.subplot(3, 3, 1 + i)
        plt.imshow(
            state[-STATE_DIM * 2 :], origin="lower", vmin=vmin, vmax=vmax, aspect="auto"
        )
        plt.colorbar()
        plt.title(state_name)

    # Shade the standard deviation of the posterior on the time series plot
    for i, idx_to_plot in enumerate(ids_to_plot):
        plt.subplot(3, 3, 7 + i)
        mean_post = posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot]
        std_post = posterior_ensemble.std(axis=(0, 2))[:, idx_to_plot]
        time_axis = np.arange(posterior_ensemble.shape[1])
        plt.fill_between(
            time_axis,
            mean_post - std_post,
            mean_post + std_post,
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        for state_at_point, state_name, color in zip(
            [
                prior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                true_sol[:, idx_to_plot],
            ],
            ["Prior Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
            ["tab:red", "tab:blue", "black"],
        ):
            plt.plot(
                state_at_point,
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.legend()
        plt.xlabel("Time")
        plt.title(f"State at grid point {idx_to_plot}")
        plt.ylim(
            true_sol[:, idx_to_plot].min()
            - np.abs(true_sol[:, idx_to_plot].min()) * 0.2,
            true_sol[:, idx_to_plot].max()
            + np.abs(true_sol[:, idx_to_plot].max()) * 0.2,
        )
        plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
