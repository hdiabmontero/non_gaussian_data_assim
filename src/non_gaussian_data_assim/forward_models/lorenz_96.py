import pdb
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.time_integrators import get_stepper


@jax.jit  # type: ignore[misc]
def L96_RHS(x: jnp.ndarray, F: float) -> jnp.ndarray:
    """Lorenz 96 right hand side."""
    return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F


class Lorenz96Model(BaseForwardModel):
    """Base class for forward models."""

    def __init__(
        self,
        forcing_term: float,
        state_dim: int,
        dt: float,
        model_integration_steps: int,
        stepper_type: str = "runge_kutta_4",
    ) -> None:
        """Initialize the forward model."""
        super().__init__(dt, model_integration_steps, state_dim)

        self.forcing_term = forcing_term
        self.num_states = 1

        self.integrator = get_stepper(stepper_type, self.dt, self.rhs)

    def rhs(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorenz 96 right hand side."""
        # Ensure x is 1D for the RHS function
        x_flat = x.flatten()
        return L96_RHS(x_flat, self.forcing_term)

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorenz 96 stepper."""
        return self.integrator(x)
