from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.jax_utils import get_pairwise_interaction_fn
from non_gaussian_data_assim.kernels import (
    get_divergence_kernel_fn,
    get_kernel_matrix_fn,
)
from non_gaussian_data_assim.localization import distance_based_localization
from non_gaussian_data_assim.observations.observation_operator import (
    LinearObservationOperator,
    NonlinearObservationOperator,
    ObservationOperator,
)
from non_gaussian_data_assim.time_integrators import get_stepper, rollout

DEFAULT_ALPHA = 0.01 / 10
DEFAULT_STEP_SIZE = 0.01 / 10
DEFAULT_B_D = 1.0


def get_prior_score_fn(
    prior_mean: np.ndarray,
    prior_cov_inv: np.ndarray,
) -> np.ndarray:
    """Get the prior score function."""

    def prior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return -prior_cov_inv @ (x_s - prior_mean)

    return prior_score_fn


def get_likelihood_score_fn_with_linear_obs_operator(
    obs_vect: np.ndarray,
    obs_matrix: np.ndarray,
    obs_cov_inv: np.ndarray,
) -> np.ndarray:
    """Get the likelihood score function."""

    def likelihood_score_fn(x_s: np.ndarray) -> np.ndarray:
        return -obs_matrix.T @ obs_cov_inv @ (obs_matrix @ x_s - obs_vect)

    return likelihood_score_fn


def get_likelihood_score_fn_with_non_linear_obs_operator(
    obs_vect: np.ndarray,
    obs_operator: NonlinearObservationOperator,
    obs_cov_inv: np.ndarray,
) -> np.ndarray:
    """Get the likelihood score function."""

    def likelihood_score_fn(x_s: np.ndarray) -> np.ndarray:
        obs_gradient = obs_operator.grad_obs_operator(x_s)
        return (
            -obs_gradient @ obs_cov_inv @ (obs_operator._obs_operator(x_s) - obs_vect)
        )

    return likelihood_score_fn


def get_posterior_score_fn(
    prior_score_fn: Callable[[np.ndarray], np.ndarray],
    likelihood_score_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Get the posterior score function."""

    def posterior_score_fn(x_s: np.ndarray) -> np.ndarray:
        return prior_score_fn(x_s) + likelihood_score_fn(x_s)

    return posterior_score_fn


def get_rhs_fn(
    posterior_score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    divergence_kernel_fn: Callable[[jnp.ndarray], jnp.ndarray],
    kernel_matrix_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the right-hand side function.

    Args:
        posterior_score_fn: Maps x_s [ensemble, dofs] -> posterior_score [ensemble, dofs].
        divergence_kernel_fn: Maps x_s -> [ensemble, ensemble, dofs].
        kernel_matrix_fn: Maps x_s -> [ensemble, ensemble, 1].
    """

    def rhs_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        """Get the right-hand side function."""

        posterior_score = posterior_score_fn(x_s)

        kernel_matrix = kernel_matrix_fn(x_s)
        divergence_kernel = divergence_kernel_fn(x_s)

        out = divergence_kernel + kernel_matrix[:, :, None] * posterior_score[None]

        out = out.sum(axis=1)

        return out / x_s.shape[0]

    return rhs_fn


class ParticleFlowFilter(BaseDataAssimilationMethod):
    def __init__(
        self,
        ensemble_size: int,
        R: np.ndarray,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        name: str = "pff",
        localization_distance: Optional[int] = None,
        num_pseudo_time_steps: int = 100,
        step_size: float = DEFAULT_STEP_SIZE,
        alpha: float = DEFAULT_ALPHA,
        kernel_type: str = "scalar",
        stepper: str = "forward_euler",
        return_pff_trajectory: bool = False,
        inflation_factor: float = 1.0,
        prior_cov_regularization: Optional[float] = None,
    ) -> None:
        """
        Initialize the Particle Flow Filter.

        Args:
            ensemble_size (int): Number of ensemble members.
            R (numpy.array): Observation error covariance matrix.
            obs_operator (Callable[[np.ndarray], np.ndarray]): Observation operator.
            forward_operator (Callable[[np.ndarray], np.ndarray]): Forward operator.
            localization_distance (int): Localization distance.
            num_pseudo_time_steps (int): Number of pseudo-time steps.
            step_size (float): Step size.
            alpha (float): Alpha parameter.
            kernel_type (str): Type of kernel to use.
            stepper (str): Type of stepper to use.
            prior_cov_regularization: If not None, add
                ``prior_cov_regularization * mean(diag(prior_cov)) * I`` to the
                localized prior covariance before inversion. Stabilizes the
                inverse when the empirical covariance is rank-deficient.
        """
        super().__init__(name, obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.dofs = self.num_states * self.state_dim
        self.num_pseudo_time_steps = num_pseudo_time_steps
        self.localization_distance = localization_distance
        self.step_size = step_size
        self.alpha = alpha
        self.kernel_type = kernel_type
        self.stepper = stepper
        self.return_pff_trajectory = return_pff_trajectory
        self.is_linear_obs_operator = obs_operator.is_linear
        self.inflation_factor = inflation_factor
        self.prior_cov_regularization = prior_cov_regularization
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
        rng_key: Optional[jax.random.PRNGKey] = None,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        x_s = prior_ensemble.reshape(self.ensemble_size, -1)  # [dofs, ensemble]

        if prior_mean is None:
            prior_mean = jnp.mean(x_s, axis=0)
        if prior_cov is None:
            prior_cov = jnp.cov(x_s.T)
        prior_cov = self.localization(prior_cov)
        prior_cov = self.inflation_factor * prior_cov
        if self.prior_cov_regularization is not None:
            eps = self.prior_cov_regularization * jnp.mean(jnp.diag(prior_cov))
            prior_cov = prior_cov + eps * jnp.eye(prior_cov.shape[0])
        prior_cov_inv = jnp.linalg.inv(prior_cov)

        if len(prior_cov.shape) == 0:
            prior_cov = prior_cov.reshape(1, 1)

        # distance_weight_matrix = jnp.linalg.inv(self.alpha * prior_cov)
        distance_weight_matrix = jnp.eye(self.state_dim) * jnp.pi

        obs_cov_inv = jnp.linalg.inv(self.R)

        # Distributions
        prior_score_fn = get_prior_score_fn(
            prior_mean=prior_mean,
            prior_cov_inv=prior_cov_inv,
        )
        if self.is_linear_obs_operator:
            likelihood_score_fn = get_likelihood_score_fn_with_linear_obs_operator(
                obs_vect=obs_vect,
                obs_matrix=self.obs_operator.obs_matrix,  # type: ignore[attr-defined]
                obs_cov_inv=obs_cov_inv,
            )
        else:
            likelihood_score_fn = get_likelihood_score_fn_with_non_linear_obs_operator(
                obs_vect=obs_vect,
                obs_operator=self.obs_operator,  # type: ignore[arg-type]
                obs_cov_inv=obs_cov_inv,
            )
        posterior_score_fn = get_posterior_score_fn(
            prior_score_fn=prior_score_fn,
            likelihood_score_fn=likelihood_score_fn,
        )
        posterior_score_vmap = jax.vmap(posterior_score_fn, in_axes=0, out_axes=0)

        # Kernels
        divergence_kernel_fn = get_pairwise_interaction_fn(
            pair_fn=get_divergence_kernel_fn(
                kernel_type=self.kernel_type,
                distance_weight_matrix=distance_weight_matrix,
            ),
        )
        kernel_matrix_fn = get_pairwise_interaction_fn(
            pair_fn=get_kernel_matrix_fn(
                kernel_type=self.kernel_type,
                distance_weight_matrix=distance_weight_matrix,
            ),
        )

        compile_rhs = True
        if compile_rhs:
            posterior_score_vmap = jax.jit(posterior_score_vmap)
            divergence_kernel_fn = jax.jit(divergence_kernel_fn)
            kernel_matrix_fn = jax.jit(kernel_matrix_fn)

        # RHS
        rhs_fn = get_rhs_fn(
            posterior_score_fn=posterior_score_vmap,
            divergence_kernel_fn=divergence_kernel_fn,
            kernel_matrix_fn=kernel_matrix_fn,
        )
        stepper = get_stepper(self.stepper, self.step_size, rhs_fn)

        rollout_fn = rollout(
            stepper,
            self.num_pseudo_time_steps,
            return_model_integration_steps=self.return_pff_trajectory,
        )
        rollout_fn = jax.jit(rollout_fn)

        x_s = rollout_fn(x_s)

        # Pseudo-time for data assimilation
        # flow = []
        # for _ in range(self.num_pseudo_time_steps):
        #     x_s = stepper(x_s)
        #     flow.append(x_s)

        # x_s = jnp.array(flow)
        # x_s = x_s[-1]

        if self.return_pff_trajectory:
            x_s = jnp.transpose(x_s, (1, 0, 2))
            return x_s.reshape(
                self.ensemble_size,
                self.num_pseudo_time_steps,
                self.num_states,
                self.state_dim,
            )

        return x_s.reshape(self.ensemble_size, self.num_states, self.state_dim)
