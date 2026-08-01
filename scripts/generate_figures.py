"""
Generate all figures for Paper 1 submission
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("/home/sentinel/sentinel_iot/paper1/results")
FIGURES = Path("/home/sentinel/sentinel_iot/paper1/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Privacy Tradeoff Curve ─────────────────────
print("Generating Figure 1: Privacy Tradeoff...")
privacy = json.load(open(str(RESULTS/"privacy_module_evaluation.json")))
tradeoff = privacy["privacy_accuracy_tradeoff"]

eps_vals  = [float(k) for k in tradeoff.keys()]
psnr_vals = [tradeoff[k]["mean_psnr_db"] for k in tradeoff]
occ_vals  = [tradeoff[k]["mean_occupancy_preservation"] for k in tradeoff]
lat_vals  = [tradeoff[k]["mean_processing_latency_ms"] for k in tradeoff]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(eps_vals, psnr_vals, "b-o", linewidth=2,
         markersize=8, label="PSNR (dB)")
ax2.plot(eps_vals, occ_vals, "r-s", linewidth=2,
         markersize=8, label="Occupancy Preservation")
ax1.axvline(x=1.0, color="green", linestyle="--",
            alpha=0.7, label="Recommended epsilon=1.0")
ax1.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
ax1.set_ylabel("PSNR (dB)", color="blue", fontsize=12)
ax2.set_ylabel("Occupancy Preservation", color="red", fontsize=12)
ax1.set_title("Privacy-Utility Tradeoff: Depth-Silhouette "
              "Anonymisation\n(Kinect v1 Model 1473)", fontsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2,
           loc="center right", fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES/"fig1_privacy_tradeoff.png"),
            dpi=300, bbox_inches="tight")
plt.savefig(str(FIGURES/"fig1_privacy_tradeoff.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved: fig1_privacy_tradeoff.png + .pdf")

# ── Figure 2: Baseline Comparison Bar Chart ───────────────
print("Generating Figure 2: Baseline Comparison...")
comp = json.load(open(str(RESULTS/"baseline_comparison.json")))
methods = [r["method"].replace(" (This Work)","*")
           for r in comp["comparison"]]
f1s  = [r["f1"]  for r in comp["comparison"]]
fprs = [r["false_positive_rate"] for r in comp["comparison"]]
drs  = [r["detection_rate"] for r in comp["comparison"]]

x = np.arange(len(methods))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, f1s,  width, label="F1 Score",
       color="steelblue", alpha=0.85)
ax.bar(x,          drs,  width, label="Detection Rate",
       color="seagreen", alpha=0.85)
ax.bar(x + width,  fprs, width, label="False Positive Rate",
       color="tomato", alpha=0.85)
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("RCAF-IoT v2* vs Baseline Methods\n"
             "(*This Work — RF+GB+IF Hybrid Ensemble)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.08)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES/"fig2_baseline_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.savefig(str(FIGURES/"fig2_baseline_comparison.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved: fig2_baseline_comparison.png + .pdf")

# ── Figure 3: Cross-Validation Box Plot ───────────────────
print("Generating Figure 3: Cross-Validation...")
cv = json.load(open(str(RESULTS/"cross_validation_results.json")))
folds = cv["fold_results"]
metrics_to_plot = ["f1", "dr", "fpr", "auc"]
labels_plot = ["F1 Score", "Detection Rate",
               "False Positive Rate", "ROC-AUC"]
data_plot = [[f[m] for f in folds] for m in metrics_to_plot]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data_plot, labels=labels_plot,
                patch_artist=True, notch=False)
colors = ["steelblue", "seagreen", "tomato", "purple"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("5-Fold Cross-Validation — RCAF-IoT v2\n"
             "(Raspberry Pi 4 ARM Cortex-A72 @ 1.8GHz)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES/"fig3_cross_validation.png"),
            dpi=300, bbox_inches="tight")
plt.savefig(str(FIGURES/"fig3_cross_validation.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved: fig3_cross_validation.png + .pdf")

# ── Figure 4: STRIDE-DREAD Risk Matrix ────────────────────
print("Generating Figure 4: STRIDE-DREAD...")
stride = json.load(open(str(RESULTS/"stride_dread_threat_model.json")))
threats = stride["threats"]
t_ids    = [t["threat_id"] for t in threats]
t_scores = [t["dread_score"] for t in threats]
t_layers = [t["layer"] for t in threats]
layer_colors = {"Device": "steelblue",
                "Network": "tomato",
                "Application": "seagreen"}
colors_bar = [layer_colors[l] for l in t_layers]

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(t_ids, t_scores, color=colors_bar, alpha=0.8)
ax.axvline(x=8.0, color="red", linestyle="--",
           linewidth=1.5, label="Critical (>=8.0)")
ax.axvline(x=6.0, color="orange", linestyle="--",
           linewidth=1.5, label="High (>=6.0)")
ax.set_xlabel("DREAD Score", fontsize=12)
ax.set_title("STRIDE-DREAD Threat Model\n"
             "Kinect v1 (Model 1473) + Raspberry Pi 4 "
             "— 19 Threat Vectors", fontsize=12)
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="steelblue", alpha=0.8, label="Device Layer"),
    Patch(facecolor="tomato",    alpha=0.8, label="Network Layer"),
    Patch(facecolor="seagreen",  alpha=0.8, label="Application Layer"),
    plt.Line2D([0],[0], color="red",    linestyle="--",
               label="Critical (>=8.0)"),
    plt.Line2D([0],[0], color="orange", linestyle="--",
               label="High (>=6.0)"),
]
ax.legend(handles=legend_els, fontsize=9, loc="lower right")
ax.set_xlim(0, 11)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES/"fig4_stride_dread.png"),
            dpi=300, bbox_inches="tight")
plt.savefig(str(FIGURES/"fig4_stride_dread.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved: fig4_stride_dread.png + .pdf")

# ── Figure 5: Latency Benchmark ───────────────────────────
print("Generating Figure 5: Latency Benchmark...")
methods_lat = ["Isolation\nForest", "One-Class\nSVM",
               "Local Outlier\nFactor", "Random\nForest",
               "Gradient\nBoosting", "RCAF-IoT v2\n(This Work)"]
latencies   = [0.014, 0.037, 0.041, 0.033, 0.008, 45.2]
colors_lat  = ["gray","gray","gray","gray","gray","steelblue"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(methods_lat, latencies, color=colors_lat, alpha=0.8)
ax.axhline(y=100, color="red", linestyle="--",
           linewidth=1.5, label="100ms real-time target")
ax.set_ylabel("Avg Latency (ms/sample)", fontsize=12)
ax.set_title("Inference Latency on Raspberry Pi 4 "
             "ARM Cortex-A72\n(lower is better — "
             "100ms = real-time threshold)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
for bar, lat in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{lat:.1f}ms", ha="center",
            va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(str(FIGURES/"fig5_latency.png"),
            dpi=300, bbox_inches="tight")
plt.savefig(str(FIGURES/"fig5_latency.pdf"),
            bbox_inches="tight")
plt.close()
print("  Saved: fig5_latency.png + .pdf")

print(f"\n{'='*50}")
print(f"ALL FIGURES GENERATED")
print(f"Location: {FIGURES}")
print(f"{'='*50}")
for f in sorted(FIGURES.glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<40} {size_kb:.0f} KB")
