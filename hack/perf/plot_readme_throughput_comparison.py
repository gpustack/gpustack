"""
Generate the throughput-comparison figure used in the top-level README.

Each bar pair compares default vLLM against the GPUStack-optimized
configuration for a recent model on NVIDIA H200 GPUs. The workload differs
per model: we use the profile that each model's Performance Lab report
recommends as its optimization target, shown under the model name.

Data source: docs/performance-lab/<model>/h200.md (Conclusion tables).
"""

import matplotlib.pyplot as plt
import numpy as np

gpu_type = "H200"

# (model label, hardware, workload profile, baseline TPS, optimized TPS)
entries = [
    ("GLM-5.2", "8xH200", "Balanced 8K/1K", 5481.90, 17525.76),
    ("DeepSeek-V3.2", "8xH200", "Medium Prompt", 10925.59, 27712.54),
    ("DeepSeek-R1", "8xH200", "Short Prompt 1K/128", 5931.25, 20448.42),
    ("Qwen3.5-35B-A3B", "1xH200", "Throughput 1K/128", 37934.72, 50464.84),
]

model_labels = [m for m, _, _, _, _ in entries]
baseline_tps = [b for _, _, _, b, _ in entries]
optimized_tps = [o for _, _, _, _, o in entries]

improvement = [(o - b) / b * 100 for b, o in zip(baseline_tps, optimized_tps)]
max_tps = max(max(baseline_tps), max(optimized_tps))
value_offset = max_tps * 0.012
improvement_offset = max_tps * 0.06

x = np.arange(len(entries))
width = 0.35

_, ax = plt.subplots(figsize=(11, 6.5))
ax.bar(x - width / 2, baseline_tps, width, label="vLLM Baseline", color="#1f77b4")
ax.bar(x + width / 2, optimized_tps, width, label="GPUStack-Optimized", color="#ff7f0e")

for i in range(len(entries)):
    ax.text(
        x[i] - width / 2,
        baseline_tps[i] + value_offset,
        f"{baseline_tps[i]:.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        x[i] + width / 2,
        optimized_tps[i] + value_offset,
        f"{optimized_tps[i]:.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        x[i] + width / 2,
        optimized_tps[i] + improvement_offset,
        f"+{improvement[i]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="green",
    )

ax.set_ylabel("Throughput (Tokens per second)")
ax.set_xlabel("Model")
ax.set_title(f"Optimizing Throughput on {gpu_type} GPUs")
ax.set_xticks(x)
ax.set_xticklabels(model_labels)
ax.legend()
plt.ylim(0, max_tps * 1.18)
plt.tight_layout()

save_path = "docs/assets/h200-throughput-comparison.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
