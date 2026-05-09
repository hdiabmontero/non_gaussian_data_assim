import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.time_integrators import get_stepper


class IdentityModel(BaseForwardModel):
    """Base class for forward models."""

    def __init__(
        self,
        state_dim: int = 2,
        rng_key: jax.random.PRNGKey = None,
    ) -> None:
        """Initialize the forward model."""
        super().__init__(
            dt=0.01, model_integration_steps=1, state_dim=state_dim, num_states=1
        )
        self.rng_key = rng_key

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Identity model one step."""
        return x
