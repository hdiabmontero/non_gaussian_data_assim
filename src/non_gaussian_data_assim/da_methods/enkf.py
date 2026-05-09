import pdb
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


class EnsembleKalmanFilter(BaseDataAssimilationMethod):
    """Ensemble Kalman Filter."""

    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        forward_operator: BaseForwardModel,
        obs_operator: ObservationOperator,
        name: str = "enkf",
        inflation_factor: float = 1.0,
        localization_distance: Optional[int] = None,
    ) -> None:
        """
        Initialize the Ensemble Kalman Filter.
        Args:
        ensemble_size (int): Number of ensemble members.
        R (numpy.array): Observation error covariance matrix.
        inflation_factor (float): Inflation factor.
        forward_operator (Callable[[np.ndarray], np.ndarray]): Forward operator.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        localization_distance (int): Localization distance.
        """
        super().__init__(name, obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.inflation_factor = inflation_factor
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.dofs = self.num_states * self.state_dim
        self.R = R
        self.localization_distance = localization_distance

        if self.localization_distance is None:
            self.localization = lambda x: x
        else:
            self.localization = lambda x: distance_based_localization(
                self.localization_distance, self.state_dim, x  # type: ignore[arg-type]
            )

    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> np.ndarray:
        """Analysis step of the Ensemble Kalman Filter.

        Args:
        mem (int): Number of ensemble members.
        nx (int): The size of the state vector.
        ensemble (numpy.array): Ensemble of state estimates.
        obs_vect (numpy.array): Observation vector.
        R (numpy.array): Observation error covariance matrix.
        rng_key (jax.random.PRNGKey): RNG key.

        Returns:
        dict: A dictionary containing the posterior ensemble, Kalman gain, innovation,
            mean and covariance of the posterior.
        """
        # Identify indices of valid observations

        # Prepare the prior state vector (ensemble matrix)
        prior_ensemble = prior_ensemble.reshape(self.ensemble_size, -1).T

        # Calculate the mean and covariance of the prior
        cov_prior = jnp.cov(prior_ensemble)
        cov_prior = self.localization(cov_prior)
        cov_prior = self.inflation_factor * cov_prior

        # Filter and perturb the observation vector
        rng_key, key = jax.random.split(rng_key)
        perturb = jax.random.multivariate_normal(
            key, jnp.zeros(self.obs_operator.num_obs), self.R, shape=(self.ensemble_size,)  # type: ignore[attr-defined]
        )
        obs_vect_perturbed = obs_vect + perturb
        obs_vect_perturbed = obs_vect_perturbed.T

        # Observation operator matrix
        obs_matrix = self.obs_operator.obs_matrix  # type: ignore[attr-defined]

        # Calculate the Kalman gain
        k_left = cov_prior @ obs_matrix.T
        k_right = obs_matrix @ cov_prior @ obs_matrix.T + self.R

        # K_left * K_right^-1
        kalman_gain = jnp.linalg.solve(k_right, k_left.T).T

        # Calculate the innovation
        innovation = obs_vect_perturbed - obs_matrix @ prior_ensemble

        # Calculate the posterior ensemble
        posterior_ensemble = prior_ensemble + kalman_gain @ innovation
        posterior_ensemble = posterior_ensemble.T
        posterior_ensemble = posterior_ensemble.reshape(
            self.ensemble_size, self.num_states, self.state_dim
        )

        return posterior_ensemble


# Compute mean and covariance of the posterior
# mean_posterior = np.mean(posterior_ensemble, axis=0)
# cov_posterior = np.cov(posterior_ensemble.T)

# Return a dictionary of EnKF outputs
# enkf_output = {
#     "posterior": posterior_vect,
#     "kalman_gain": kalman_gain,
#     "innovation": innovation,
#     "mean_post": mean_posterior,
#     "cov_post": cov_posterior,
# }
