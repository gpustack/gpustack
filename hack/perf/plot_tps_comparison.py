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

    x = np.arange(len(case_names))
    width = 0.35

    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_tps, width, label='vLLM Baseline')
    ax.bar(x + width / 2, optimized_tps, width, label='Optimized')

    # Add text annotations
    for i in range(len(case_names)):
        # TPS values
        ax.text(
            x[i] - width / 2,
            baseline_tps[i] + 100,
            f'{baseline_tps[i]:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
        ax.text(
            x[i] + width / 2,
            optimized_tps[i] + 100,
            f'{optimized_tps[i]:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )
        # Improvement percentage
        offset = 500
        if improvement[i] > 0:
            ax.text(
                x[i] + width / 2,
                optimized_tps[i] + offset,
                f'(+{improvement[i]:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                color='green',
            )
        else:
            ax.text(
                x[i] + width / 2,
                optimized_tps[i] + offset,
                f'({improvement[i]:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                color='red',
            )

    ax.set_ylabel(ylabel)
    ax.set_title(f'Optimizing {model_name} Throughput on {gpu_type} GPUs')
    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.legend()
    plt.ylim(0, max(optimized_tps) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save to file if path is given
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


# Example usage
model_name = "gpt-oss-120b"
gpu_type = "A100"
case_names = [
    "ShareGPT",
    "Short Prompt",
    "Medium Prompt",
    "Long Prompt",
    "Very Long Prompt",
]
baseline_tps = [4768.75, 9353.12, 8456.65, 7671.00, 4337.21]
optimized_tps = [5055.82, 10785.60, 9249.89, 8468.21, 4605.91]

plot_throughput_comparison(
    model_name,
    gpu_type,
    case_names,
    baseline_tps,
    optimized_tps,
    save_path="throughput_comparison.png",
)
