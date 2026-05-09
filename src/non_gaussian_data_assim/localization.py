import jax.numpy as jnp


def distance_based_localization(
    r_influ: int, state_dim: int, cov_prior: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply localization to the covariance matrix.

    Args:
    r_influ (int): The radius of influence for localization -- grid cells.
    state_dim (int): The dimension of the state vector.
    cov_prior (jax.numpy.array): The prior covariance matrix.

    Returns:
    jax.numpy.array: Localized covariance matrix.
    """
    # Create a localization mask with Gaussian-like decay
    tmp = jnp.zeros((state_dim, state_dim))
    for i in range(1, 3 * r_influ + 1):
        tmp += jnp.exp(-(i**2) / r_influ**2) * (
            jnp.diag(jnp.ones(state_dim - i), i) + jnp.diag(jnp.ones(state_dim - i), -i)
        )
    mask = tmp + jnp.diag(jnp.ones(state_dim))

    # Apply the localization mask to the prior covariance matrix
    cov_prior_loc = jnp.zeros(cov_prior.shape)
    for i in range(1, 2):
        for j in range(1, 2):
            cov_prior_loc = cov_prior_loc.at[
                (i - 1) * state_dim : i * state_dim, (j - 1) * state_dim : j * state_dim
            ].set(
                jnp.multiply(
                    cov_prior[
                        (i - 1) * state_dim : i * state_dim,
                        (j - 1) * state_dim : j * state_dim,
                    ],
                    mask,
                )
            )

    return cov_prior_loc
