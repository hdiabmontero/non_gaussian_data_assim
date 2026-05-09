import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.jax_utils import get_pairwise_interaction_fn


def exp_scalar_kernel_fn(
    x: jnp.ndarray,
    y: jnp.ndarray,
    distance_weight_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Get the scalar kernel function."""

    # dx = x-y
    # sqdist = jnp.sum(dx**2)
    # kernel = jnp.exp(-0.5 * sqdist * distance_weight_matrix)

    # return kernel

    return jnp.exp(-0.5 * (x - y).T @ distance_weight_matrix @ (x - y))


def get_kernel_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the kernel function."""

    if kernel_type == "scalar":
        return functools.partial(exp_scalar_kernel_fn, **kwargs)
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_kernel_matrix_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the kernel matrix function."""

    if kernel_type == "scalar":
        scalar_kernel_fn = get_kernel_fn(kernel_type, **kwargs)

        def scalar_kernel_matrix_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Calculate the scalar kernel matrix."""
            return scalar_kernel_fn(x, y)  # * jnp.eye(x.shape[0])

        return scalar_kernel_matrix_fn
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_divergence_kernel_fn(
    kernel_type: str,
    **kwargs: Any,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Get the divergence of the kernel function."""

    if kernel_type == "scalar":
        scalar_kernel_fn = get_kernel_fn(kernel_type, **kwargs)
        distance_weight_matrix = kwargs.get("distance_weight_matrix")

        def divergence_scalar_kernel_matrix_fn(
            x: jnp.ndarray, y: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculate the divergence of the scalar kernel matrix."""
            return -distance_weight_matrix.T @ (x - y) * scalar_kernel_fn(x, y)  # type: ignore[union-attr]

        return divergence_scalar_kernel_matrix_fn
    else:
        raise ValueError(
            f"Invalid kernel type: {kernel_type}. We only support 'scalar' kernel type."
        )


def get_pairwise_interactions_fn(
    kernel_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get the pairwise interactions function.

    Returns a function that takes x_s [ensemble, dofs] and returns
    [ensemble, ensemble, *output_shape] where result[i, j, ...] = kernel_fn(x_s[j], x_s[i]).
    """
    return get_pairwise_interaction_fn(kernel_fn)


def get_pairwise_kernel_scalar_fn(
    kernel_matrix_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Get pairwise kernel function returning [ensemble, ensemble, 1].

    Extracts the scalar k(x,y) from kernel_matrix_fn (which returns k*I for scalar kernel).
    """

    def scalar_pair_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        K = kernel_matrix_fn(x, y)
        return jnp.expand_dims(K[0, 0], axis=-1)

    return get_pairwise_interaction_fn(scalar_pair_fn)
