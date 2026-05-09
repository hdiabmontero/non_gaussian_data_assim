"""Plotting helpers for DA experiments."""

from typing import Mapping, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_low_dim_trajectory(
    *,
    true_sol: jnp.ndarray,
    reference_ensemble: jnp.ndarray,
    posterior_ensemble: jnp.ndarray,
    title: str,
    da_method_name: str,
    ensemble_size: int,
    reference_metrics: Mapping[str, float],
    posterior_metrics: Mapping[str, float],
    state_dim: int,
    state_names: Sequence[str],
    data_assimilation_steps: int,
    model_integration_steps: int,
) -> None:
    """Per-state-dimension subplots for low-dimensional systems (e.g. Lorenz 63)."""
    true_sol_2d = true_sol.reshape(
        data_assimilation_steps * model_integration_steps + 1, state_dim
    )
    mean_reference = reference_ensemble.mean(axis=(0, 2))
    mean_post = posterior_ensemble.mean(axis=(0, 2))
    std_post = posterior_ensemble.std(axis=(0, 2))
    time_axis = np.arange(posterior_ensemble.shape[1])

    plt.figure()
    plt.suptitle(
        f"{title}, DA Method: {da_method_name}, Ensemble Size: {ensemble_size}, \n"
        f"Reference RMSE: {reference_metrics['rmse']:.4f}, "
        f"Posterior RMSE: {posterior_metrics['rmse']:.4f}"
    )
    for state_idx in range(state_dim):
        plt.subplot(state_dim, 1, state_idx + 1)
        for state_name, state_data, color in zip(
            ["Reference Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
            [mean_reference, mean_post, true_sol_2d],
            ["tab:blue", "tab:red", "black"],
        ):
            plt.plot(
                time_axis,
                state_data[:, state_idx],
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.fill_between(
            time_axis,
            mean_post[:, state_idx] - std_post[:, state_idx],
            mean_post[:, state_idx] + std_post[:, state_idx],
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(state_names[state_idx])
        plt.ylim(true_sol_2d[:, state_idx].min(), true_sol_2d[:, state_idx].max())
    plt.show()


def plot_high_dim_field(
    *,
    true_sol: jnp.ndarray,
    reference_ensemble: jnp.ndarray,
    posterior_ensemble: jnp.ndarray,
    title: str,
    da_method_name: str,
    ensemble_size: int,
    reference_metrics: Mapping[str, float],
    posterior_metrics: Mapping[str, float],
    state_dim: int,
    data_assimilation_steps: int,
    model_integration_steps: int,
    state_names: Optional[Sequence[str]] = None,
) -> None:
    """Spatial-field heatmaps + per-point time series for high-dim systems (L96, KS)."""
    del state_names
    true_sol_2d = true_sol.reshape(
        data_assimilation_steps * model_integration_steps + 1, state_dim
    )
    ids_to_plot = [state_dim // 4, state_dim // 2, 3 * state_dim // 4]

    fields = [
        true_sol_2d,
        reference_ensemble.mean(axis=(0, 2)),
        posterior_ensemble.mean(axis=(0, 2)),
        true_sol_2d - reference_ensemble.mean(axis=(0, 2)),
        true_sol_2d - posterior_ensemble.mean(axis=(0, 2)),
        posterior_ensemble.var(axis=(0, 2)),
    ]
    field_names = [
        "True Solution",
        "Reference Ensemble Mean",
        "Posterior Ensemble Mean",
        "|True - Reference| difference",
        "|True - Posterior| difference",
        "Posterior Ensemble Variance",
    ]

    plt.figure()
    plt.suptitle(
        f"{title}, DA Method: {da_method_name}, Ensemble Size: {ensemble_size}, \n"
        f"Reference RMSE: {reference_metrics['rmse']:.4f}, "
        f"Posterior RMSE: {posterior_metrics['rmse']:.4f}"
    )

    for i, (field, field_name) in enumerate(zip(fields, field_names)):
        vmin = true_sol_2d.min() if i < 3 else np.percentile(field, 5)
        vmax = true_sol_2d.max() if i < 3 else np.percentile(field, 95)
        plt.subplot(3, 3, 1 + i)
        plt.imshow(
            field[-state_dim * 2 :],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar()
        plt.title(field_name)

    for i, idx_to_plot in enumerate(ids_to_plot):
        plt.subplot(3, 3, 7 + i)
        mean_post_pt = posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot]
        std_post_pt = posterior_ensemble.std(axis=(0, 2))[:, idx_to_plot]
        time_axis = np.arange(posterior_ensemble.shape[1])
        plt.fill_between(
            time_axis,
            mean_post_pt - std_post_pt,
            mean_post_pt + std_post_pt,
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        for state_at_point, state_name, color in zip(
            [
                reference_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
                true_sol_2d[:, idx_to_plot],
            ],
            ["Reference Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
            ["tab:red", "tab:blue", "black"],
        ):
            plt.plot(
                state_at_point,
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.legend()
        plt.xlabel("Time")
        plt.title(f"State at grid point {idx_to_plot}")
        plt.ylim(
            true_sol_2d[:, idx_to_plot].min()
            - np.abs(true_sol_2d[:, idx_to_plot].min()) * 0.2,
            true_sol_2d[:, idx_to_plot].max()
            + np.abs(true_sol_2d[:, idx_to_plot].max()) * 0.2,
        )
        plt.grid(True)
    plt.show()
