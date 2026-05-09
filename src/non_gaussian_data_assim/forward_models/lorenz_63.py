import pdb
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.time_integrators import get_stepper


def L63_RHS(x: jnp.ndarray, sigma: float, beta: float, rho: float) -> jnp.ndarray:
    """Lorenz 96 right hand side."""
    return jnp.array(
        [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    )


class Lorenz63Model(BaseForwardModel):
    """Base class for forward models."""

    def __init__(
        self,
        dt: float,
        model_integration_steps: int,
        sigma: float = 10.0,
        beta: float = 2.6666666,
        rho: float = 28.0,
        stepper_type: str = "runge_kutta_4",
    ) -> None:
        """Initialize the forward model."""
        super().__init__(dt, model_integration_steps, 3)

        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.num_states = 1

        self.integrator = get_stepper(stepper_type, self.dt, self.rhs)

    def rhs(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorenz 96 right hand side."""
        # Ensure x is 1D for the RHS function
        x_flat = x.flatten()
        return L63_RHS(x_flat, self.sigma, self.beta, self.rho)

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lorenz 96 stepper."""
        return self.integrator(x)
