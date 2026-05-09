import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel


class KuramotoSivashinsky(BaseForwardModel):
    """Kuramoto Sivashinsky forward model."""

    def __init__(
        self,
        dt: float,
        model_integration_steps: int,
        state_dim: int,
        domain_length: float,
    ):
        """Initialize the Kuramoto Sivashinsky forward model.

        Args:
            dt: Time step for integration
            num_model_steps: Number of inner time steps per call
            state_dim: Number of spatial grid points
            domain_length: Length of the periodic domain
        """
        super().__init__(dt, model_integration_steps, state_dim)

        self.L = domain_length
        self.state_dim = state_dim
        self.domain_length = domain_length

        wavenumbers = jnp.fft.rfftfreq(
            state_dim, d=domain_length / (state_dim * 2 * jnp.pi)
        )
        self.derivative_operator = 1j * wavenumbers

        linear_operator = -self.derivative_operator**2 - self.derivative_operator**4
        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )

        self.alias_mask = wavenumbers < 2 / 3 * jnp.max(wavenumbers)

    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Kuramoto Sivashinsky one step."""
        u_nonlin = -0.5 * x**2
        u_hat = jnp.fft.rfft(x)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_next_hat = self.exp_term * u_hat + self.coef * u_nonlin_der_hat
        u_next = jnp.fft.irfft(u_next_hat, n=self.state_dim)
        return u_next
