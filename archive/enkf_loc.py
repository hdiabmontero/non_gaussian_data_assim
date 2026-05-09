from typing import Any, Callable, Dict

import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.localization import localization
from non_gaussian_data_assim.observation_operator import ObservationOperator


def enkf(
    mem: int,
    nx: int,
    ensemble: np.ndarray,
    obs_vect: np.ndarray,
    R: np.ndarray,
    N: int,
    r_influ: int,
) -> Dict[str, Any]:
    """
    Implement the Ensemble Kalman Filter with localization.
    Args:
    mem (int): Number of ensemble members.
    nx (int): Size of the state vector.
    ensemble (numpy.array): Ensemble of state estimates.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.
    N (int): The number of grid points.
    r_influ (int): Radius of influence for localization -- grid cells.
    Returns:
    dict: Posterior ensemble, Kalman gain, innovation, mean and covariance of the posterior.
    """
    # Extract indices of valid observations and set up the prior state vector
    index_obs = np.where(obs_vect > -999)[0]
    prior_vect = ensemble

    # Compute mean and covariance of the prior, and apply localization
    mean_prior = np.mean(prior_vect, axis=1)
    cov_prior = np.cov(prior_vect)
    cov_prior = localization(r_influ, N, cov_prior)

    # Perturb the observation vector
    obs_vect_filtered = obs_vect[index_obs]
    obs_vect_perturbed = np.zeros((len(index_obs), mem))
    for i in range(mem):
        obs_vect_perturbed[:, i] = obs_vect_filtered

    # Set up the observation error covariance matrix and compute H matrix
    cov_obs = R[:, :]
    h_matrix = h_operator(nx, obs_vect)

    # Calculate the Kalman gain
    k_left = cov_prior.dot(h_matrix.T)
    k_right_inv = np.linalg.inv(h_matrix.dot(cov_prior).dot(h_matrix.T) + cov_obs)
    kalman_gain = k_left.dot(k_right_inv)

    # Calculate the innovation and update the posterior ensemble
    innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)
    posterior_vect = prior_vect + kalman_gain.dot(innovation)

    # Compute mean and covariance of the posterior
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)

    # Output dictionary
    enkf_output = {
        "posterior": posterior_vect,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "mean_post": mean_posterior,
        "cov_post": cov_posterior,
    }
    return enkf_output


class EnsembleKalmanFilterLocalization(BaseDataAssimilationMethod):
    def __init__(
        self,
        mem: int,
        nx: int,
        R: np.ndarray,
        N: int,
        r_influ: int,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
    ) -> None:
        """
        Implement the Ensemble Kalman Filter with localization.
        Args:
        mem (int): Number of ensemble members.
        nx (int): Size of the state vector.
        ensemble (numpy.array): Ensemble of state estimates.
        obs_vect (numpy.array): Observation vector.
        R (numpy.array): Observation error covariance matrix.
        N (int): The number of grid points.
        r_influ (int): Radius of influence for localization -- grid cells.
        obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
        """
        super().__init__(obs_operator)
        self.mem = mem
        self.nx = nx
        self.R = R
        self.N = N
        self.r_influ = r_influ

    def _assimilate_data(
        self, prior_ensemble: np.ndarray, obs_vect: np.ndarray
    ) -> np.ndarray:
        """Assimilate the data."""
        return enkf(
            mem=self.mem,
            nx=self.nx,
            ensemble=prior_ensemble,
            obs_vect=obs_vect,
            R=self.R,
            N=self.N,
            r_influ=self.r_influ,
        )["posterior"]
