import pdb
from typing import Callable

import jax
import jax.numpy as jnp


def rollout(
    stepper: Callable,
    num_steps: int,
    return_model_integration_steps: bool = False,
    include_initial_state: bool = False,
) -> jnp.ndarray:
    """Rollout the system using the given stepper."""

    def scan_fn(x: jnp.ndarray, _: None = None) -> jnp.ndarray:
        """Scan function for the rollout."""
        next_x = stepper(x)
        return next_x, next_x

    def rollout_fn(init: jnp.ndarray) -> jnp.ndarray:
        """Rollout function for the rollout."""
        last_step, trajectory = jax.lax.scan(scan_fn, init, xs=None, length=num_steps)

        if return_model_integration_steps:
            if include_initial_state:
                trajectory = jnp.concatenate([init[None, ...], trajectory], axis=0)

            return trajectory
        else:
            return last_step

    return rollout_fn


def rollout_with_model_integration_steps(
    inner_rollout_fn: Callable,
    data_assimilation_steps: int,
    include_initial_state: bool = False,
) -> jnp.ndarray:
    """Rollout the system using the given stepper with inner steps."""

    def scan_fn(x: jnp.ndarray, _: jnp.ndarray) -> jnp.ndarray:
        """Scan function for the rollout."""
        trajectory = inner_rollout_fn(x)
        return trajectory[-1], trajectory

    def rollout_fn(init: jnp.ndarray) -> jnp.ndarray:
        """Rollout function for the rollout."""
        state_dim = init.shape[-1]
        num_states = init.shape[-2]

        _, trajectory = jax.lax.scan(
            scan_fn,
            init,
            # (init, init),
            xs=None,
            length=data_assimilation_steps,
        )
        trajectory = trajectory.reshape(-1, num_states, state_dim)

        return jnp.concatenate([init[None, :, :], trajectory], axis=0)

    return rollout_fn


def get_runge_kutta_4(dt: float, rhs: Callable) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the Runge-Kutta 4th order time integrator."""

    def stepper(x: jnp.ndarray) -> jnp.ndarray:
        """Runge-Kutta 4th order time integration."""
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return stepper


def get_forward_euler(dt: float, rhs: Callable) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the Forward Euler time integrator."""

    def stepper(x: jnp.ndarray) -> jnp.ndarray:
        """Forward Euler time integration."""
        return x + dt * rhs(x)

    return stepper


def get_backward_euler(
    dt: float, rhs: Callable
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the Backward Euler time integrator using automatic differentiation."""

    def stepper(x: jnp.ndarray) -> jnp.ndarray:
        """Backward Euler time integration.

        Solves the implicit equation: x_{n+1} = x_n + dt * rhs(x_{n+1})
        using Newton's method with automatic differentiation.
        """
        # Store original shape
        original_shape = x.shape

        # Initial guess: use forward Euler
        x_new = x + dt * rhs(x)

        # Flatten for Newton's method
        x_flat = x.flatten()
        x_new_flat = x_new.flatten()

        # Residual function: F(x_new) = x_new - x_old - dt * rhs(x_new)
        # Works with flattened arrays
        def residual(x_new_flat: jnp.ndarray) -> jnp.ndarray:
            x_new = x_new_flat.reshape(original_shape)
            rhs_val = rhs(x_new)
            res = x_new - x - dt * rhs_val
            return res.flatten()

        # Newton's method with automatic differentiation
        # Compute Jacobian using automatic differentiation
        jacobian_fn = jax.jacfwd(residual)

        # Newton iterations using JAX control flow
        max_iter = 10
        tol = 1e-10

        # State: (x_new_flat, iter_count)
        def cond_fn(state: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            """Continue if not converged and under max iterations."""
            x_new_flat, iter_count = state
            res = residual(x_new_flat)
            not_converged = jnp.linalg.norm(res) >= tol
            return jnp.logical_and(not_converged, iter_count < max_iter)

        def body_fn(
            state: tuple[jnp.ndarray, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Perform one Newton iteration."""
            x_new_flat, iter_count = state
            res = residual(x_new_flat)
            jac = jacobian_fn(x_new_flat)
            # Solve: J * delta = -residual
            delta = jnp.linalg.solve(jac, -res)
            x_new_flat = x_new_flat + delta
            return x_new_flat, iter_count + 1

        # Initial state
        initial_state = (x_new_flat, jnp.array(0))

        # Run Newton's method
        x_new_flat, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)

        # Reshape back to original shape
        return x_new_flat.reshape(original_shape)

    return stepper


def get_stepper(
    stepper_type: str, dt: float, rhs: Callable
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the stepper function."""
    if stepper_type == "runge_kutta_4":
        return get_runge_kutta_4(dt, rhs)
    elif stepper_type == "forward_euler":
        return get_forward_euler(dt, rhs)
    elif stepper_type == "backward_euler":
        return get_backward_euler(dt, rhs)
    else:
        raise ValueError(
            f"Invalid stepper type: {stepper_type}. We only support 'runge_kutta_4', 'forward_euler', and 'backward_euler'."
        )


# class RungeKutta4:
#     """Runge-Kutta 4th order time integrator."""

#     def __init__(self, dt: float, rhs: Callable):
#         """Initialize the Runge-Kutta 4th order time integrator."""
#         self.dt = dt
#         self.rhs = rhs

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """Runge-Kutta 4th order time integration."""
#         k1 = self.rhs(x)
#         k2 = self.rhs(x + 0.5 * self.dt * k1)
#         k3 = self.rhs(x + 0.5 * self.dt * k2)
#         k4 = self.rhs(x + self.dt * k3)
#         return x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# class ForwardEuler:
#     """Forward Euler time integrator."""

#     def __init__(self, dt: float, rhs: Callable):
#         """Initialize the Forward Euler time integrator."""
#         self.dt = dt
#         self.rhs = rhs

#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """Forward Euler time integration."""
#         return x + self.dt * self.rhs(x)
