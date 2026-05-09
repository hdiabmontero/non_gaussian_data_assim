import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.time_integrators import get_stepper


class SineModel(BaseForwardModel):
    """Base class for forward models."""

    def __init__(
        self,
    ) -> None:
        """Initialize the forward model."""
        super().__init__(dt=0.01, model_integration_steps=1, state_dim=2, num_states=1)

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sine model one step."""
        return jnp.array([[1.0 + jnp.sin(jnp.pi * x[0, 0]) + x[0, 1], x[0, 0]]])
