import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import gaussian_kde
from scipy.io import loadmat
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.forward_models.sine import SineModel
from non_gaussian_data_assim.observation_operator import (
    LinearObservationOperator,
    SineObservationOperator,
    SineObservationOperatorNoError,
)

jax.config.update("jax_disable_jit", False)

SEED = 42

ENSEMBLE_SIZE = 200

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
        "return_pff_trajectory": True,
        "num_pseudo_time_steps": 10000,
        "step_size": 1 / 100,
        # "stepper": "runge_kutta_4",
        "stepper": "forward_euler",
        # "stepper": "backward_euler",
    },
}

NUM_STATES = 1
STATE_DIM = 1

# Observation ids
OBS_IDS = (0,)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 1.0
B = 1.0


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Define the forward model
    forward_model = IdentityModel(state_dim=STATE_DIM)

    # Define the observation operator
    obs_operator = SineObservationOperatorNoError(
        obs_states=OBS_STATES,
        obs_indices=OBS_IDS,
        state_dim=STATE_DIM,
    )

    # Generate observations
    observations = jnp.array([0.0])

    da_model = DA_METHODS[DA_METHOD](
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        **SPECIFIC_DA_ARGS[DA_METHOD],
    )

    # Initialize the prior ensemble
    prior_ensemble = jnp.linspace(-1.0, 3.0, ENSEMBLE_SIZE).reshape(ENSEMBLE_SIZE, 1, 1)
    prior_kde = gaussian_kde(prior_ensemble[:, 0, 0])

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    t0 = time.time()
    posterior_ensemble = da_model(
        prior_ensemble=posterior_ensemble[:, -1],
        obs_vect=observations,
        return_model_integration_steps=False,
        prior_mean=jnp.ones(1),
        prior_cov=jnp.eye(1),
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")

    posterior_ensemble = posterior_ensemble[:, -1, 0, 0]
    kde = gaussian_kde(posterior_ensemble)

    x = np.linspace(-2.5, 5.0, 100)

    benjamin_samples = loadmat("benjamin_case/testcaseA_det.mat")
    X_PFF = benjamin_samples["X_PFF"][:, 0, -1]
    pdf_PFF = benjamin_samples["pdf_PFF"].flatten()
    pdf_SDE = benjamin_samples["pdf_SDE"].flatten()
    x_pdf_PFF = benjamin_samples["x_pdf_PFF"].flatten()
    x_pdf_SDE = benjamin_samples["x_pdf_SDE"].flatten()

    n_bins = 50

    plt.figure()
    plt.plot(x, prior_kde.pdf(x), color="tab:orange", linewidth=3, label="Prior")
    plt.hist(X_PFF, bins=n_bins, density=True, color="tab:red", alpha=0.2)
    plt.plot(x_pdf_PFF, pdf_PFF, color="tab:red", linewidth=3, label="Benjamin PFF")
    plt.plot(x_pdf_SDE, pdf_SDE, color="tab:blue", linewidth=3, label="Benjamin SDE")
    plt.hist(
        posterior_ensemble, bins=n_bins, density=True, color="tab:green", alpha=0.2
    )
    plt.plot(x, kde.pdf(x), color="tab:green", linewidth=3, label="Nikolaj PFF")
    plt.grid(True)
    plt.xlim(x.min(), x.max())
    plt.legend()
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
