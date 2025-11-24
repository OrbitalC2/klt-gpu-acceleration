from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent

datasets = ["Small", "Medium", "Large"]
versions = ["CPU", "D2", "D3", "D4"]
labels = {
    "CPU": "CPU Baseline",
    "D2": "Initial CUDA Port",
    "D3": "Optimized CUDA",
    "D4": "OpenACC Directives",
}
times = {
    "Small": {"CPU": 0.287, "D2": 0.106, "D3": 0.070, "D4": 0.219},
    "Medium": {"CPU": 0.838, "D2": 0.192, "D3": 0.145, "D4": 0.357},
    "Large": {"CPU": 3.310, "D2": 0.460, "D3": 0.346, "D4": 0.871},
}
speedups = {
    dataset: {version: times[dataset]["CPU"] / times[dataset][version] for version in versions}
    for dataset in datasets
}

colors = {
    "CPU": "#475569",
    "D2": "#2563EB",
    "D3": "#16A34A",
    "D4": "#F97316",
}


def style_axes(ax, title, ylabel=None):
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#334155")


def savefig(name):
    plt.savefig(OUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def grouped_speedup_chart():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(datasets))
    width = 0.19
    plot_versions = versions

    for idx, version in enumerate(plot_versions):
        values = [speedups[d][version] for d in datasets]
        bars = ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=labels[version],
            color=colors[version],
            edgecolor="white",
            linewidth=1.2,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0F172A",
            )

    style_axes(ax, "KLT Tracker Speedup by Dataset", "Speedup vs CPU")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 10.7)
    ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.text(
        0.99,
        -0.14,
        "Best result: optimized CUDA reaches 9.57x on the large dataset",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color="#475569",
    )
    savefig("klt_speedup_by_dataset")


def runtime_chart():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(datasets))
    width = 0.19

    for idx, version in enumerate(versions):
        values = [times[d][version] for d in datasets]
        bars = ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=labels[version],
            color=colors[version],
            edgecolor="white",
            linewidth=1.1,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.045,
                f"{value:.3f}s",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=0,
                color="#0F172A",
            )

    style_axes(ax, "Execution Time Across Implementations", "Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 3.75)
    ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    savefig("klt_runtime_by_dataset")


def speedup_matrix_chart():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    matrix = np.array([[speedups[d][version] for d in datasets] for version in versions])
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=1.0, vmax=10.0, aspect="auto")

    ax.set_title("Speedup Matrix by Implementation and Dataset", fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(len(versions)))
    ax.set_yticklabels([labels[v] for v in versions], fontsize=11)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, colors="#334155")
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_idx, version in enumerate(versions):
        for col_idx, dataset in enumerate(datasets):
            value = speedups[dataset][version]
            text_color = "white" if value >= 5.5 else "#0F172A"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}x",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup vs CPU", rotation=270, labelpad=18)
    savefig("klt_speedup_matrix")


def time_saved_chart():
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(datasets))
    width = 0.24
    plot_versions = ["D2", "D3", "D4"]

    for idx, version in enumerate(plot_versions):
        values = [
            (times[d]["CPU"] - times[d][version]) / times[d]["CPU"] * 100.0
            for d in datasets
        ]
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=labels[version],
            color=colors[version],
            edgecolor="white",
            linewidth=1.2,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0F172A",
            )

    style_axes(ax, "Runtime Reduction vs CPU", "Time saved (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    savefig("klt_runtime_reduction_percent")


def summary_dashboard():
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.38)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(datasets))
    width = 0.19
    for idx, version in enumerate(versions):
        values = [speedups[d][version] for d in datasets]
        bars = ax1.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=labels[version],
            color=colors[version],
            edgecolor="white",
            linewidth=1.1,
        )
        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.16,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color="#0F172A",
            )
    style_axes(ax1, "Speedup vs CPU", "Speedup")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontweight="bold")
    ax1.set_ylim(0, 10.9)
    ax1.legend(frameon=False, loc="upper left", fontsize=10)

    large_times = [times["Large"][v] for v in versions]
    ax2.barh([labels[v] for v in versions], large_times, color=[colors[v] for v in versions])
    ax2.invert_yaxis()
    style_axes(ax2, "Large Dataset Runtime")
    ax2.set_xlabel("Seconds", fontsize=10)
    for i, value in enumerate(large_times):
        ax2.text(value + 0.03, i, f"{value:.3f}s", va="center", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(0, 3.7)

    matrix_versions = ["D2", "D3", "D4"]
    matrix = np.array([[speedups[d][version] for d in datasets] for version in matrix_versions])
    ax3.imshow(matrix, cmap="YlGnBu", vmin=1.0, vmax=10.0, aspect="auto")
    ax3.set_title("Categorical Speedup Matrix", fontsize=17, fontweight="bold", pad=14)
    ax3.set_xticks(np.arange(len(datasets)))
    ax3.set_xticklabels(datasets, fontweight="bold")
    ax3.set_yticks(np.arange(len(matrix_versions)))
    ax3.set_yticklabels([labels[v] for v in matrix_versions], fontsize=9)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(colors="#334155")
    for row_idx, version in enumerate(matrix_versions):
        for col_idx, dataset in enumerate(datasets):
            value = speedups[dataset][version]
            text_color = "white" if value >= 5.5 else "#0F172A"
            ax3.text(
                col_idx,
                row_idx,
                f"{value:.2f}x",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    fig.suptitle("KLT Feature Tracker GPU Acceleration", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Optimized CUDA: 0.346s on large dataset vs 3.310s CPU baseline (9.57x speedup)",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    savefig("klt_performance_dashboard")


def main():
    grouped_speedup_chart()
    runtime_chart()
    speedup_matrix_chart()
    time_saved_chart()
    summary_dashboard()
    print(f"Generated graphs in {OUT_DIR}")


if __name__ == "__main__":
    main()
