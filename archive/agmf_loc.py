from typing import Any, Dict

import numpy as np

from non_gaussian_data_assim.gaussian_mixture import gaussian_mixt
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observation_operator import h_operator
from non_gaussian_data_assim.rand_utils import randsample


def agmf(
    mem: int,
    nx: int,
    ensemble: np.ndarray,
    obs_vect: np.ndarray,
    R: np.ndarray,
    h: float,
    w_prev: np.ndarray,
    nc_threshold: float,
    N: int,
    r_influ: int,
) -> Dict[str, Any]:
    """
    Implement the Adaptive Gaussian Mixture Filter.

    Args:
    mem (int): Number of ensemble members.
    nx (int): Size of the state vector.
    ensemble (numpy.array): Ensemble of state estimates.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.
    h (float): Scale factor for covariance matrix.
    w_prev (numpy.array): Previous weights of the ensemble members.
    N (int): The number of grid points.
    nc_threshold (float): Threshold for deciding whether resampling is necessary.
    r_influ (int): Radius of influence for localization -- grid cells.

    Returns:
    dict: Dictionary containing the posterior ensemble, Kalman gain, innovation,
          mean and covariance of the posterior, weights, and alpha value.
    """
    # Identifying indices of valid observations
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)

    # Preparing the prior state vector (ensemble matrix)
    prior_vect = ensemble

    # Calculating the mean and covariance of the prior
    mean_prior = np.mean(prior_vect, axis=1)
    cov_prior = (h**2) * np.cov(prior_vect)

    # Apply localization to the prior covariance matrix
    cov_prior = localization(r_influ, N, cov_prior)

    # Filtering and perturbing the observation vector
    obs_vect_filtered = obs_vect[index_obs]
    obs_vect_perturbed = np.zeros((num_obs, mem))
    for i in range(mem):
        for j in range(num_obs):
            obs_vect_perturbed[j, i] = obs_vect_filtered[j]

    # Observation error covariance matrix
    cov_obs = R[:, :]

    # Calculating the observation operator matrix
    h_matrix = h_operator(nx, obs_vect)

    # Calculating the Kalman gain
    k_left = cov_prior.dot(np.transpose(h_matrix))
    k_right_inv = np.linalg.inv(
        h_matrix.dot(cov_prior).dot(np.transpose(h_matrix)) + cov_obs
    )
    kalman_gain = k_left.dot(k_right_inv)

    # Calculating the innovation
    innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)

    # Calculating the posterior ensemble
    posterior_vect = prior_vect + kalman_gain.dot(innovation)
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)

    # Recalculating weights
    w_t = gaussian_mixt(
        w_prev, num_obs, posterior_vect, obs_vect_perturbed, h_matrix, R
    )

    # Evaluating degeneracy and calculating the bridging alpha
    N_eff = 1 / np.sum(w_t**2)
    alpha = N_eff / mem

    # Adjusting weights
    w_t = w_t * alpha + (1 - alpha) * (1 / mem)

    # Resampling if necessary
    resamp = 0
    if N_eff < nc_threshold:
        J = randsample(mem, w_t)
        epsc = np.random.normal(0, 0.1, mem)
        for i in range(mem):
            posterior_vect[:, i] = (
                posterior_vect[:, int(J[i])] + np.sqrt(np.diag(cov_posterior)) * epsc[i]
            )
        cov_posterior = (h**2) * np.cov(posterior_vect)
        resamp = 1

    # Result output
    agmf_output = {
        "posterior": posterior_vect,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "mean_post": mean_posterior,
        "cov_post": cov_posterior,
        "weights": w_t,
        "alpha": alpha,
    }

    return agmf_output
