from matplotlib.transforms import ScaledTranslation
import polars as pl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from pathlib import Path


def load_sample_trajectories():
    return {
        "s_traj": pl.read_csv("data/example_traj_s.csv"),
        "x_traj": pl.read_csv("data/example_traj_x.csv"),
    }


def load_trajectory_data():
    """Load data for trajectory length comparison."""
    return {
        "ground_truth": pl.read_csv("experiments/pws/result.csv"),
        "mlpws_result": pl.read_csv("experiments/mlpws/result.csv"),
        "doe_result": pl.read_csv("experiments/doe/result.csv"),
        "infonce_result": pl.read_csv("experiments/infonce/result.csv"),
    }


def load_gain_sweep_data():
    """Load and process gain sweep data."""
    result_dir = Path("experiments/gain_sweep01")
    data = []
    for p in result_dir.glob("*"):
        if not p.is_dir():
            continue
        with open(p / "parameters.json", "r") as file:
            params = json.load(file)
        result_path = p / "result.csv"
        result = pl.read_csv(result_path)
        data.append(
            {
                "gain": params["gain"],
                "estimator": params["estimator"],
                "MI": result["mean"][-1],
                "sample_size": params["num_pairs"],
                "result_path": params["result_path"],
            }
        )
    data = pl.DataFrame(data)

    return {
        "ground_truth": data.filter(pl.col("estimator") == "PWS").sort("gain"),
        "ml_pws": data.filter(pl.col("estimator") == "ML-PWS").sort("gain"),
        "gaussian1": data.filter(
            (pl.col("estimator") == "Gaussian") & (pl.col("sample_size") == 1000)
        ).sort("gain"),
        "gaussian2": data.filter(
            (pl.col("estimator") == "Gaussian") & (pl.col("sample_size") == 100000)
        ).sort("gain"),
    }


def setup_plot_style():
    """Configure the plot style."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 8,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0.0,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.5,
        }
    )
    return 510 / 72, 160 / 72  # width, height


def plot_sample_trajectories(axs, data):
    test_s = data["s_traj"]["s"]
    time = data["s_traj"]["time"]
    axs[0, 0].plot(time, test_s, color="#008F00")
    axs[0, 0].text(
        0.05,
        0.95,
        "$s(t)$",
        fontsize=8,
        transform=axs[0, 0].transAxes,
        verticalalignment="top",
        color="#008F00",
    )
    axs[0, 0].set_yticks([-2, 0, 2])
    axs[0, 0].set_ylim(-3, 3)
    for i in [0, 1]:
        axs[0, i].set_xticklabels([])
        axs[i, 1].set_yticklabels([])
        axs[1, i].set_xlabel("$t$")
    for ax in axs.reshape(-1):
        ax.grid(visible=True)
        ax.set_xlim(1, 50)
        ax.tick_params(axis="both", which="major", pad=3)
    for gain, coord in zip([0.1, 1.0, 10.0], [(0, 1), (1, 0), (1, 1)]):
        axs[coord].text(
            0.05,
            0.95,
            f"$x(t)$",
            transform=axs[coord].transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            color="#0096FF",
        )
        axs[coord].text(
            0.98,
            0.95,
            f"$g = {gain}$",
            transform=axs[coord].transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
        )
        x_traj = data["x_traj"].filter(pl.col("gain") == gain)
        axs[coord].plot(time, x_traj["mean"], color="#0096FF")
        axs[coord].fill_between(
            time, x_traj["q10"], x_traj["q90"], alpha=0.25, color="#0096FF"
        )
        axs[coord].set_ylim(-0.3, 1.9)


def plot_trajectory_comparison(ax, data):
    """Plot trajectory length comparison (right subplot)."""
    colors = plt.get_cmap("Dark2").colors
    ax.set_prop_cycle(plt.cycler(color=colors))

    # Plot ground truth
    ax.plot(
        data["ground_truth"]["step"] + 1,
        data["ground_truth"]["mean"] / np.log(2),
        label="ground truth (PWS)",
        color="black",
        zorder=3,
    )

    # Plot ML-PWS
    ax.plot(
        data["mlpws_result"]["step"],
        data["mlpws_result"]["mean"] / np.log(2),
        "*",
        label="ML-PWS",
        zorder=2,
    )

    # Plot DoE
    ax.plot(
        data["doe_result"]["step"],
        data["doe_result"]["mean"] / np.log(2),
        "^",
        label="DoE",
        zorder=1,
    )

    # Plot InfoNCE
    ax.plot(
        data["infonce_result"]["step"],
        data["infonce_result"]["mean"] / np.log(2),
        "x",
        markeredgewidth=1.5,
        label="InfoNCE",
    )

    # Add reference line and text
    ax.axhline(np.log2(1000), color="#A6A3CE", linestyle="--", zorder=0)
    ax.text(
        1,
        np.log2(1000) + 0.5,
        "$\\log_2(N)$",
        horizontalalignment="left",
        verticalalignment="bottom",
        color="#5B55A0",
    )

    # Configure axis
    ax.legend(frameon=False)
    ax.set_xlabel("trajectory length")
    ax.set_ylabel("mutual information (bit)")
    ax.set_xlim(0, 50)
    ax.set_ylim(0)


def plot_gain_comparison(ax, data):
    """Plot gain comparison (left subplot)."""
    colors = plt.get_cmap("Dark2").colors
    labels = ["ground truth (PWS)", "ML-PWS", "Gaussian I", "Gaussian II"]
    formats = ["-", "*", ":", "--"]
    datasets = [
        data["ground_truth"],
        data["ml_pws"],
        data["gaussian1"],
        data["gaussian2"],
    ]
    plot_colors = ["black", colors[0], colors[1], colors[1]]

    ax.set_xscale("log")
    for label, df, fmt, c in zip(labels, datasets, formats, plot_colors):
        ax.plot(df["gain"], df["MI"] / np.log(2), fmt, color=c, label=label)

    ax.legend(frameon=False)
    ax.set_ylabel("mutual information (bit)")
    ax.set_xlabel("gain $\\gamma$")
    ax.set_xlim(0.025, 20)
    ax.set_ylim(0, 60)


def main():
    """Main function to create and save the figure."""
    # Load data
    sample_trajectory_data = load_sample_trajectories()
    trajectory_data = load_trajectory_data()
    gain_sweep_data = load_gain_sweep_data()

    # Setup plot
    width, height = setup_plot_style()
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1, 1, 1],
        left=0.04,
        right=0.98,
        top=0.9,
        bottom=0.25,
        wspace=0.35,
        hspace=0.0,
    )

    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.1
    )
    ax11 = fig.add_subplot(gs_left[0, 0])
    ax12 = fig.add_subplot(gs_left[0, 1])
    ax21 = fig.add_subplot(gs_left[1, 0])
    ax22 = fig.add_subplot(gs_left[1, 1])
    axs_col1 = np.array([[ax11, ax12], [ax21, ax22]])

    # ---- Column 2: single axis ----
    ax_col2 = fig.add_subplot(gs[0, 1])

    # ---- Column 3: single axis ----
    ax_col3 = fig.add_subplot(gs[0, 2])

    ax11.text(
        0.0,
        1.0,
        "a",
        transform=(
            ax11.transAxes + ScaledTranslation(-0 / 72, 3 / 72, fig.dpi_scale_trans)
        ),
        fontsize=10,
        va="bottom",
        fontfamily="sans-serif",
        weight="bold",
    )
    ax_col2.text(
        0.0,
        1.0,
        "b",
        transform=(
            ax_col2.transAxes + ScaledTranslation(-0 / 72, 3 / 72, fig.dpi_scale_trans)
        ),
        fontsize=10,
        va="bottom",
        fontfamily="sans-serif",
        weight="bold",
    )
    ax_col3.text(
        0.0,
        1.0,
        "c",
        transform=(
            ax_col3.transAxes + ScaledTranslation(-0 / 72, 3 / 72, fig.dpi_scale_trans)
        ),
        fontsize=10,
        va="bottom",
        fontfamily="sans-serif",
        weight="bold",
    )

    # Create plots
    plot_sample_trajectories(axs_col1, sample_trajectory_data)
    plot_gain_comparison(ax_col2, gain_sweep_data)
    plot_trajectory_comparison(ax_col3, trajectory_data)

    # Save figure
    fig.savefig("reports/figures/ml_pws_fig_1.pdf")
    fig.savefig("reports/figures/ml_pws_fig_1.png")
    return fig


if __name__ == "__main__":
    main()
