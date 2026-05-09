import pdb
from abc import abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from numpy.typing import NDArray


class ObservationOperator:
    """Observation operator for selecting specific states and indices from a 3D state array."""

    def __init__(
        self,
        is_linear: bool = True,
    ):
        self.is_linear = is_linear

    @abstractmethod
    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to a single state."""
        raise NotImplementedError

    @abstractmethod
    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        raise NotImplementedError

    def __call__(self, ensemble: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to the ensemble."""
        return jax.vmap(self._obs_operator)(ensemble)


class LinearObservationOperator(ObservationOperator):
    """Observation operator for selecting specific states and indices from a 3D state array.

    The state array is expected to have shape [ensemble_size, num_states, state_dim].
    This operator selects:
    - Specific states from the num_states dimension (via obs_states)
    - Specific indices from the state_dim dimension (via obs_indices)

    The output is flattened to shape [ensemble_size, len(obs_states) * len(obs_indices)].
    """

    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: np.ndarray,
        state_dim: int,
    ):
        """
        Initialize the observation operator.

        Args:
            obs_states: Tuple of state indices to observe (from num_states dimension).
                       Example: (0, 1) to observe states at indices 0 and 1.
            obs_indices: Array of indices to observe (from state_dim dimension).
                        Example: np.arange(0, 50, 5) to observe indices 0, 5, 10, ..., 45.
        """
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = len(obs_states)
        self.num_obs = len(obs_indices) * len(obs_states)

        self.is_linear = True

        self.obs_matrix = get_obs_matrix(
            obs_states, obs_indices, self.num_states, self.state_dim
        )
        self.obs_matrix = sparse.BCOO.fromdense(self.obs_matrix)

    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        return self.obs_matrix.T

    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to a single state."""
        return self.obs_matrix @ x.flatten()


def get_obs_matrix(
    obs_states: np.ndarray,
    obs_indices: np.ndarray,
    num_states: int,
    state_dim: int,
) -> np.ndarray:
    """
    Create an observation operator matrix H that maps flattened state to observations.

    The matrix H has shape [num_states * state_dim, len(obs_states) * len(obs_indices)].
    When multiplied with a flattened state array of shape [..., num_states * state_dim],
    it produces the same output as the ObservationOperator applied to the unflattened state.

    Args:
        obs_states: Array of state indices to observe (from num_states dimension).
                   Example: np.array([0, 1]) to observe states at indices 0 and 1.
        obs_indices: Array of indices to observe (from state_dim dimension).
                    Example: np.arange(0, 50, 5) to observe indices 0, 5, 10, ..., 45.
        num_states: Total number of states in the state array.
        state_dim: Dimension of each state.

    Returns:
        Observation matrix of shape [num_states * state_dim, len(obs_states) * len(obs_indices)].
        The output observations are ordered as: all obs_indices for state0, then all obs_indices for state1, etc.
    """
    obs_states = np.asarray(obs_states)
    obs_indices = np.asarray(obs_indices)

    num_obs_states = len(obs_states)
    num_obs_indices = len(obs_indices)
    total_obs = num_obs_states * num_obs_indices
    total_state_size = num_states * state_dim

    # Initialize the observation matrix
    H = np.zeros((total_state_size, total_obs))

    # For each observed state and each observed index, set the corresponding entry to 1
    obs_col = 0
    for state_idx in obs_states:
        for dim_idx in obs_indices:
            # Calculate the flattened index in the state array
            # Flattening order: state0_dim0, state0_dim1, ..., state0_dimN, state1_dim0, ...
            flattened_idx = state_idx * state_dim + dim_idx
            H[flattened_idx, obs_col] = 1
            obs_col += 1

    return H.T


class NonlinearObservationOperator(ObservationOperator):
    """Observation operator"""

    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: np.ndarray,
        state_dim: int,
        nonlinear_fn: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        """Initialize the observation operator."""
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = len(obs_states)
        self.num_obs = len(obs_indices) * len(obs_states)

        self.is_linear = False

        self.nonlinear_fn = nonlinear_fn

    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        return jax.grad(self._obs_operator)(x)

    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.nonlinear_fn(x)


class SineObservationOperator(NonlinearObservationOperator):
    """Observation operator"""

    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: np.ndarray,
        state_dim: int,
    ):
        """Initialize the observation operator."""
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = len(obs_states)
        self.num_obs = len(obs_indices) * len(obs_states)

        self.is_linear = False

    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        return jnp.array([jnp.pi * jnp.cos(jnp.pi * x[0]), 1.0])

    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:

        return 1.0 + jnp.sin(jnp.pi * x[0]) + x[1]


class SineObservationOperatorNoError(NonlinearObservationOperator):
    """Observation operator"""

    def __init__(
        self,
        obs_states: np.ndarray,
        obs_indices: np.ndarray,
        state_dim: int,
    ):
        """Initialize the observation operator."""
        self.obs_states = obs_states
        self.obs_indices = obs_indices
        self.state_dim = state_dim
        self.num_states = len(obs_states)
        self.num_obs = len(obs_indices) * len(obs_states)
        self.is_linear = False

    def grad_obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the observation operator."""
        return jnp.pi * jnp.cos(jnp.pi * x)

    def _obs_operator(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the observation operator to a single state."""
        return 1.0 + jnp.sin(jnp.pi * x)
