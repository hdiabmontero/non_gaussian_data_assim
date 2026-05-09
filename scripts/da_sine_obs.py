import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.forward_models.sine import SineModel
from non_gaussian_data_assim.observation_operator import (
    LinearObservationOperator,
    SineObservationOperator,
)

jax.config.update("jax_disable_jit", False)

SEED = 42

ENSEMBLE_SIZE = 1000

DA_METHOD = "pff"
DA_METHODS = {
    "enkf": EnsembleKalmanFilter,
    "agmf": AdaptiveGaussianMixtureFilter,
    "pff": ParticleFlowFilter,
}
SPECIFIC_DA_ARGS = {
    "enkf": {
        "inflation_factor": 5.0,
    },
    "agmf": {
        "inflation_factor": 1.0,
        "nc_threshold": 0.5,
        "w_prev": np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    },
    "pff": {
        "num_pseudo_time_steps": 2000,
        "step_size": 0.05,
        # "stepper": "runge_kutta_4",
        "stepper": "forward_euler",
    },
}

NUM_STATES = 1
STATE_DIM = 2

# Observation ids
OBS_IDS = (0,)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 1.0


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Define the forward model
    rng_key, key = jax.random.split(rng_key)
    # forward_model = SineModel()
    forward_model = IdentityModel(state_dim=STATE_DIM)

    # Define the observation operator
    rng_key, key = jax.random.split(rng_key)
    obs_operator = SineObservationOperator(
        obs_states=OBS_STATES,
        obs_indices=OBS_IDS,
        state_dim=STATE_DIM,
    )
    # obs_operator = LinearObservationOperator(
    #     obs_states=OBS_STATES,
    #     obs_indices=OBS_IDS,
    #     state_dim=STATE_DIM,
    # )

    # Generate observations
    rng_key, key = jax.random.split(rng_key)
    observations = jnp.array(
        [[0.0]]
    )  # jax.random.normal(key, (ENSEMBLE_SIZE, len(OBS_IDS))) * np.sqrt(R)
    # observations = jnp.linspace(-3.0, 3.0, 100).reshape(100, 1)

    da_model = DA_METHODS[DA_METHOD](
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        **SPECIFIC_DA_ARGS[DA_METHOD],
    )

    # Initialize the prior ensemble
    prior_ensemble = jnp.zeros((ENSEMBLE_SIZE, NUM_STATES, STATE_DIM))
    rng_key, key = jax.random.split(rng_key)
    # prior_ensemble = prior_ensemble.at[:, 0, 0].set(jax.random.normal(key, (ENSEMBLE_SIZE,))) + 1.0
    prior_ensemble = prior_ensemble.at[:, 0, 0].set(
        jnp.linspace(-1.0, 3.0, ENSEMBLE_SIZE)
    )
    rng_key, key = jax.random.split(rng_key)
    prior_ensemble = prior_ensemble.at[:, 0, 1].set(
        jax.random.normal(key, (ENSEMBLE_SIZE,)) * 0.25
    )

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    rng_key, key = jax.random.split(rng_key)
    posterior_ensemble = da_model(
        prior_ensemble=posterior_ensemble[:, -1],
        obs_vect=observations,
        rng_key=key,
        return_model_integration_steps=True,
    )

    posterior_ensemble = posterior_ensemble[:, 0, 0, 0]
    kde = gaussian_kde(posterior_ensemble)

    x = np.linspace(-1.5, 3.0, 100)

    plt.hist(posterior_ensemble, bins=100, density=True, color="tab:blue", alpha=0.2)
    plt.plot(x, kde.pdf(x), color="tab:blue", linewidth=5)
    plt.grid(True)
    plt.xlim(-1.5, 3.0)
    plt.xlabel("x")
    plt.ylabel("p(x|y)")
    plt.title("Posterior Distribution")
    plt.show()

    # pdb.set_trace()

    # # Calculate the prior and posterior errors
    # true_sol = true_sol.reshape(DATA_ASSIMILATION_STEPS * MODEL_INTEGRATION_STEPS + 1, STATE_DIM)

    # # Calculate the prior and posterior errors
    # prior_error = true_sol - prior_ensemble.mean(axis=(0, 2))
    # prior_error = np.sqrt(np.sum(prior_error**2))
    # posterior_error = true_sol - posterior_ensemble.mean(axis=(0, 2))
    # posterior_error = np.sqrt(np.sum(posterior_error**2))

    # print(f"Prior error: {prior_error}")
    # print(f"Posterior error: {posterior_error}")

    # mean_prior = prior_ensemble.mean(axis=(0, 2))
    # mean_post = posterior_ensemble.mean(axis=(0, 2))
    # std_post = posterior_ensemble.std(axis=(0, 2))
    # time_axis = np.arange(posterior_ensemble.shape[1])

    # state_names = ["x", "y", "z"]

    # plt.figure()
    # plt.suptitle(f"Lorenz 63, DA Method: {DA_METHOD}, Ensemble Size: {ENSEMBLE_SIZE}, \n Prior Error: {prior_error:.2f}, Posterior Error: {posterior_error:.2f}")
    # for state_idx in range(STATE_DIM):
    #     plt.subplot(STATE_DIM, 1, state_idx + 1)
    #     for i, (state_name, state_data, color) in enumerate(
    #         zip(
    #             ["Prior Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
    #             [mean_prior, mean_post, true_sol],
    #             ["tab:red", "tab:blue", "black"],
    #         )
    #     ):
    #         plt.plot(
    #             time_axis,
    #             state_data[:, state_idx],
    #             label=state_name,
    #             color=color,
    #             linewidth=3,
    #             linestyle="--" if state_name == "True Solution" else "-",
    #         )
    #     plt.fill_between(
    #         time_axis,
    #         mean_post[:, state_idx] - std_post[:, state_idx],
    #         mean_post[:, state_idx] + std_post[:, state_idx],
    #         color="tab:blue",
    #         alpha=0.2,
    #         label="Posterior ± Std",
    #     )
    #     plt.legend()
    #     plt.xlabel("Time")
    #     plt.ylabel(f"{state_names[state_idx]}")
    #     plt.ylim(true_sol[:, state_idx].min(), true_sol[:, state_idx].max())
    # plt.show()


if __name__ == "__main__":
    main()
