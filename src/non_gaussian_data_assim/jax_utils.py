from typing import Callable

import jax
import jax.numpy as jnp


def get_pairwise_interaction_fn(
    pair_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create a vectorized function that computes pairwise interactions over an ensemble.

    For pair_fn(x, y) with x, y of shape [dofs], returns a function that takes
    x_s of shape [ensemble, dofs] and returns an array of shape
    [ensemble, ensemble, *output_shape] where result[i, j, ...] = pair_fn(x_s[j], x_s[i]).

    The first dimension (i) corresponds to the outer loop (anchor particle).
    The second dimension (j) corresponds to the inner loop (interacting particle).

    Args:
        pair_fn: Function f(x, y) taking two [dofs] vectors, returns an array of any shape.

    Returns:
        A function that takes x_s [ensemble, dofs] and returns [ensemble, ensemble, *output_shape].
    """

    def pairwise_interaction_fn(x_s: jnp.ndarray) -> jnp.ndarray:
        """
        Compute pairwise interactions.

        Args:
            x_s: State array of shape [ensemble, dofs].

        Returns:
            Array of shape [ensemble, ensemble, *output_shape] where
            result[i, j, ...] = pair_fn(x_s[j, :], x_s[i, :]).
        """
        # Inner vmap over j: for fixed i, compute pair_fn(x_s[j, :], x_s[i, :]) for all j
        # in_axes=(0, None): batch over rows (axis 0) of first arg, second arg is broadcast
        inner_vmap = jax.vmap(pair_fn, in_axes=(0, None), out_axes=0)

        # Outer vmap over i: for each i, run inner_vmap over all j
        # in_axes=(None, 0): batch over rows of second arg (x_s[i] for each i)
        vmap_func = jax.vmap(inner_vmap, in_axes=(None, 0), out_axes=0)

        return vmap_func(x_s, x_s)

    return pairwise_interaction_fn
