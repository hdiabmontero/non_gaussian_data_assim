import functools
import pdb
from abc import abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.time_integrators import (
    rollout,
    rollout_with_model_integration_steps,
)


class BaseForwardModel:
    """Base class for forward models."""

    def __init__(
        self,
        dt: float,
        model_integration_steps: int,
        state_dim: int,
        num_states: int = 1,
    ):
        """Initialize the forward model."""
        self.num_states = num_states
        self.state_dim = state_dim
        self.dt = dt
        self.model_integration_steps = model_integration_steps

    @abstractmethod
    def one_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute one inner step of the forward model."""
        raise NotImplementedError

    def __call__(
        self,
        x: jnp.ndarray,
        _: None = None,
        return_model_integration_steps: bool = False,
        is_ensemble: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward the model.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]

        Returns:
            tuple: (new_time, new_state) where new_state has shape [ensemble, num_states, state_dim]
        """

        rollout_fn = rollout(
            self.one_step,
            self.model_integration_steps,
            return_model_integration_steps=return_model_integration_steps,
            include_initial_state=not return_model_integration_steps,
        )
        if is_ensemble:
            rollout_fn = jax.vmap(rollout_fn)
        rollout_fn = jax.jit(rollout_fn)
        return rollout_fn(x)

    def rollout(
        self,
        x: jnp.ndarray,
        data_assimilation_steps: int,
        return_model_integration_steps: bool = False,
    ) -> jnp.ndarray:
        """
        Outer rollout the model for the given number of outer steps.

        Args:
            x: State array of shape [ensemble, num_states, state_dim]
            data_assimilation_steps: Number of outer steps to rollout

        Returns:
            State array of shape [ensemble, num_states, state_dim]
        """

        if return_model_integration_steps:
            inner_rollout_fn = functools.partial(
                self.__call__,
                return_model_integration_steps=True,
            )
            rollout_fn = rollout_with_model_integration_steps(
                inner_rollout_fn,
                data_assimilation_steps,
                include_initial_state=False,
            )
        else:
            rollout_fn = rollout(
                self.__call__,
                data_assimilation_steps,
                return_model_integration_steps=True,
                include_initial_state=True,
            )
        rollout_fn = jax.jit(rollout_fn)
        return jax.vmap(rollout_fn)(x)
