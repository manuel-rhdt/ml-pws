import polars as pl
from matplotlib import pyplot as plt
import numpy as np
import json
from pathlib import Path


def load_trajectory_data():
    """Load data for trajectory length comparison."""
    return {
        'ground_truth': pl.read_csv('experiments/pws/result.csv'),
        'mlpws_result': pl.read_csv('experiments/mlpws/result.csv'),
        'doe_result': pl.read_csv('experiments/doe/result.csv'),
        'infonce_result': pl.read_csv('experiments/infonce/result.csv')
    }


def load_gain_sweep_data():
    """Load and process gain sweep data."""
    result_dir = Path('experiments/gain_sweep01')
    data = []
    for p in result_dir.glob('*'):
        if not p.is_dir():
            continue
        with open(p / 'parameters.json', "r") as file:
            params = json.load(file)
        result_path = p / 'result.csv'
        result = pl.read_csv(result_path)
        data.append({
            "gain": params['gain'],
            'estimator': params['estimator'],
            'MI': result['mean'][-1],
            'sample_size': params['num_pairs'],
            'result_path': params['result_path'],
        })
    data = pl.DataFrame(data)
    
    return {
        'ground_truth': data.filter(pl.col("estimator") == "PWS").sort("gain"),
        'ml_pws': data.filter(pl.col("estimator") == "ML-PWS").sort("gain"),
        'gaussian1': data.filter(
            (pl.col("estimator") == "Gaussian") & (pl.col("sample_size") == 1000)
        ).sort("gain"),
        'gaussian2': data.filter(
            (pl.col("estimator") == "Gaussian") & (pl.col("sample_size") == 100000)
        ).sort("gain")
    }


def setup_plot_style():
    """Configure the plot style."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 8,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.0,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.5,
    })
    return 340 / 72, 160 / 72  # width, height


def plot_trajectory_comparison(ax, data):
    """Plot trajectory length comparison (right subplot)."""
    colors = plt.get_cmap("Dark2").colors
    ax.set_prop_cycle(plt.cycler(color=colors))
    
    # Plot ground truth
    ax.plot(
        data['ground_truth']["step"] + 1,
        data['ground_truth']["mean"] / np.log(2),
        label="ground truth (PWS)",
        color="black",
        zorder=3,
    )
    
    # Plot ML-PWS
    ax.plot(
        data['mlpws_result']["step"],
        data['mlpws_result']["mean"] / np.log(2),
        "*",
        label="ML-PWS",
        zorder=2,
    )
    
    # Plot DoE
    ax.plot(
        data['doe_result']["step"],
        data['doe_result']["mean"] / np.log(2),
        "^",
        label="DoE",
        zorder=1,
    )
    
    # Plot InfoNCE
    ax.plot(
        data['infonce_result']["step"],
        data['infonce_result']["mean"] / np.log(2),
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
    datasets = [data['ground_truth'], data['ml_pws'], data['gaussian1'], data['gaussian2']]
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
    trajectory_data = load_trajectory_data()
    gain_sweep_data = load_gain_sweep_data()
    
    # Setup plot
    width, height = setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)
    
    # Create plots
    plot_gain_comparison(ax1, gain_sweep_data)
    plot_trajectory_comparison(ax2, trajectory_data)
    
    # Save figure
    fig.savefig('reports/figures/ml_pws_comparisons.pdf')
    fig.savefig('reports/figures/ml_pws_comparisons.png')
    return fig


if __name__ == "__main__":
    main()