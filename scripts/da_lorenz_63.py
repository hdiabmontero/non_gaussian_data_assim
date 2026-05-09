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

DATA_ASSIMILATION_STEPS = 100
MODEL_INTEGRATION_STEPS = 50
ENSEMBLE_SIZE = 100

DA_METHOD = "enkf"


DA_METHODS = {
    "enkf": EnsembleKalmanFilter,
    "agmf": AdaptiveGaussianMixtureFilter,
    "pff": ParticleFlowFilter,
}
SPECIFIC_DA_ARGS = {
    "enkf": {
        "inflation_factor": 1.0,
    },
    "agmf": {
        "inflation_factor": 1.0,
        "nc_threshold": 0.5,
        "w_prev": np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    },
    "pff": {
        "num_pseudo_time_steps": 1500,
        "step_size": 0.05,
        "stepper": "runge_kutta_4",
    },
}

# Constants and parameters
DT = 0.01
SIGMA = 10.0
BETA = 2.6666666
RHO = 28.0
T0 = 0.0

NUM_STATES = 1
STATE_DIM = 3

# Observation ids
OBS_IDS = np.arange(0, 3)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 5.0


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Initial state - shape: [1, num_states, state_dim] for single ensemble member
    rng_key, key = jax.random.split(rng_key)
    X_0 = jax.random.normal(key, (1, NUM_STATES, STATE_DIM)) * 10

    # Define the forward model
    forward_model = Lorenz63Model(
        dt=DT,
        model_integration_steps=MODEL_INTEGRATION_STEPS,
        sigma=SIGMA,
        beta=BETA,
        rho=RHO,
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

    print_metrics_table(prior_error, posterior_error, title="Lorenz 63 Metrics")

    true_sol = true_sol.reshape(
        DATA_ASSIMILATION_STEPS * MODEL_INTEGRATION_STEPS + 1, STATE_DIM
    )
    mean_prior = prior_ensemble.mean(axis=(0, 2))
    mean_post = posterior_ensemble.mean(axis=(0, 2))
    std_post = posterior_ensemble.std(axis=(0, 2))
    time_axis = np.arange(posterior_ensemble.shape[1])

    state_names = ["x", "y", "z"]

    plt.figure()
    plt.suptitle(
        f"Lorenz 63, DA Method: {DA_METHOD}, Ensemble Size: {ENSEMBLE_SIZE}, \n Prior RMSE: {prior_error['rmse']:.4f}, Posterior RMSE: {posterior_error['rmse']:.4f}"
    )
    for state_idx in range(STATE_DIM):
        plt.subplot(STATE_DIM, 1, state_idx + 1)
        for i, (state_name, state_data, color) in enumerate(
            zip(
                ["Prior Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
                [mean_prior, mean_post, true_sol],
                ["tab:red", "tab:blue", "black"],
            )
        ):
            plt.plot(
                time_axis,
                state_data[:, state_idx],
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.fill_between(
            time_axis,
            mean_post[:, state_idx] - std_post[:, state_idx],
            mean_post[:, state_idx] + std_post[:, state_idx],
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(f"{state_names[state_idx]}")
        plt.ylim(true_sol[:, state_idx].min(), true_sol[:, state_idx].max())
    plt.show()


if __name__ == "__main__":
    main()
