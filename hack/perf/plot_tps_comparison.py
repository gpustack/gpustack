"""
Plot throughput comparison between baseline and optimized versions.
Adjust the example data as needed.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_throughput_comparison(
    model_name,
    gpu_type,
    case_names,
    baseline_tps,
    optimized_tps,
    optimized_gpu_ratio=1,
    save_path=None,
):
    """
    Plot throughput comparison between baseline and optimized versions.

    Args:
        model_name (str): Name of the model.
        gpu_type (str): Type of GPU.
        case_names (list of str): List of test case names.
        baseline_tps (list of float): Throughput of baseline.
        optimized_tps (list of float): Throughput of optimized version.
        optimized_gpu_ratio (float): Normalization factor for GPUs. Default is 1.
        save_path (str, optional): If given, save plot to this file path.
    """
    # Normalize optimized TPS if needed
    optimized_tps = [x * optimized_gpu_ratio for x in optimized_tps]

    # Label for y-axis
    ylabel = "Throughput (TPS)"
    if optimized_gpu_ratio != 1:
        ylabel = "Throughput (TPS, normalized by GPU count)"

    # Compute improvements
    improvement = [(o - b) / b * 100 for b, o in zip(baseline_tps, optimized_tps)]
    max_tps = max(max(baseline_tps), max(optimized_tps))
    value_offset = max_tps * 0.015
    improvement_offset = max_tps * 0.08

    x = np.arange(len(case_names))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_tps, width, label='vLLM Baseline')
    ax.bar(x + width / 2, optimized_tps, width, label='GPUStack-Optimized')

    # Add text annotations
    for i in range(len(case_names)):
        # TPS values
        ax.text(
            x[i] - width / 2,
            baseline_tps[i] + value_offset,
            f'{baseline_tps[i]:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
        ax.text(
            x[i] + width / 2,
            optimized_tps[i] + value_offset,
            f'{optimized_tps[i]:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
        # Improvement percentage
        if improvement[i] > 0:
            ax.text(
                x[i] + width / 2,
                optimized_tps[i] + improvement_offset,
                f'(+{improvement[i]:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                color='green',
            )
        else:
            ax.text(
                x[i] + width / 2,
                optimized_tps[i] + improvement_offset,
                f'({improvement[i]:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                color='red',
            )

    ax.set_ylabel(ylabel)
    ax.set_title(
        f'{model_name} Throughput on {gpu_type} GPUs: vLLM Baseline vs. GPUStack-Optimized'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    plt.xticks(rotation=30, ha='right')

    ax.legend()
    plt.ylim(0, max_tps * 1.18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save to file if path is given
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


# Example usage
model_name = "Qwen3.5-350B-A3B"
gpu_type = "H200"
case_names = [
    "ShareGPT",
    "Input 1024 / Output 128 (Profile: Throughput)",
    "Input 32000 / Output 100 (Profile: Long Context)",
    "Input 1000 / Output 2000 (Profile: Generation Heavy)",
]
baseline_tps = [9632.01, 37934.72, 44993.20, 10455.38]
optimized_tps = [10570.88, 50464.84, 56424.42, 12258.79]

plot_throughput_comparison(
    model_name,
    gpu_type,
    case_names,
    baseline_tps,
    optimized_tps,
    save_path="throughput_comparison.png",
)
