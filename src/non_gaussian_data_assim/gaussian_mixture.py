import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def gaussian_mixt(
    weight_vect: np.ndarray,
    n_obs: int,
    ens_vect: np.ndarray,
    obs_vect: np.ndarray,
    h_matrix: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the weights for ensemble members using Gaussian Mixture Model.

    Args:
    weight_vect (numpy.array): Current weights of the ensemble members.
    n_obs (int): Number of observations.
    ens_vect (numpy.array): Ensemble matrix.
    obs_vect (numpy.array): Observation vector.
    h_matrix (numpy.array): Observation operator matrix.
    cov_matrix (numpy.array): Covariance matrix of the observations.

    Returns:
    numpy.array: Updated weights for the ensemble members.
    """
    # Normalizing factor for Gaussian probability density function
    norm_factor = 1 / jnp.sqrt(((2 * np.pi) ** n_obs) * jnp.linalg.det(cov_matrix))
    weight_mixt = jnp.zeros(len(weight_vect))
    prob_dens = jnp.zeros(len(weight_vect))

    # Calculating the weights based on the Gaussian distribution
    for i in range(ens_vect.shape[1]):
        innovation = obs_vect[:, 0] - h_matrix @ ens_vect[:, i]
        prob_dens = prob_dens.at[i].set(
            norm_factor
            * jnp.exp(
                -(1 / 2) * ((innovation.T @ jnp.linalg.inv(cov_matrix) @ innovation))
            )
        )
        weight_mixt = weight_mixt.at[i].set(prob_dens[i] * weight_vect[i])

    # Normalizing the weights
    weight_final = weight_mixt / jnp.sum(weight_mixt)

    return weight_final
