"""
Create comprehensive visualizations for the Non-linear Diction research.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"
PLOTS_DIR = f"{RESULTS_DIR}/plots"

# Load results
with open(f"{RESULTS_DIR}/probing_results.json") as f:
    probing = json.load(f)

with open(f"{RESULTS_DIR}/best_layers.json") as f:
    best_layers = json.load(f)

with open(f"{RESULTS_DIR}/prompting_results.json") as f:
    prompting = json.load(f)

with open(f"{RESULTS_DIR}/metadata.json") as f:
    metadata = json.load(f)

import os
steering = None
if os.path.exists(f"{RESULTS_DIR}/steering_results.json"):
    with open(f"{RESULTS_DIR}/steering_results.json") as f:
        steering = json.load(f)

STYLES = [s for s in probing.keys() if s != "neutral"]
LAYERS = metadata["probe_layers"]

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# ============================================================
# FIGURE 1: Linear vs MLP probe accuracy per style (best layer)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(STYLES))
width = 0.35

linear_accs = [probing[s][str(best_layers[s])]["linear_acc"] for s in STYLES]
mlp_accs = [probing[s][str(best_layers[s])]["mlp_acc"] for s in STYLES]

bars1 = ax.bar(x - width/2, linear_accs, width, label="Linear Probe", color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width/2, mlp_accs, width, label="MLP Probe", color="#DD8452", alpha=0.85)

ax.set_ylabel("Accuracy (5-fold CV)")
ax.set_xlabel("Style / Emotion")
ax.set_title("Linear vs MLP Probe Accuracy by Style (Best Layer)\nAll styles show NL index ≈ 1.0: representations are linear")
ax.set_xticks(x)
ax.set_xticklabels(STYLES, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0.7, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_linear_vs_mlp_probes.png", bbox_inches="tight")
plt.close()
print("Saved: 01_linear_vs_mlp_probes.png")


# ============================================================
# FIGURE 2: Layer-wise probing accuracy heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Linear probe heatmap
linear_matrix = np.array([
    [probing[s][str(l)]["linear_acc"] for l in LAYERS] for s in STYLES
])
sns.heatmap(linear_matrix, ax=axes[0], annot=True, fmt=".3f",
            xticklabels=LAYERS, yticklabels=STYLES,
            cmap="YlOrRd", vmin=0.6, vmax=1.0)
axes[0].set_title("Linear Probe Accuracy by Style × Layer")
axes[0].set_xlabel("Layer")

# MLP probe heatmap
mlp_matrix = np.array([
    [probing[s][str(l)]["mlp_acc"] for l in LAYERS] for s in STYLES
])
sns.heatmap(mlp_matrix, ax=axes[1], annot=True, fmt=".3f",
            xticklabels=LAYERS, yticklabels=STYLES,
            cmap="YlOrRd", vmin=0.6, vmax=1.0)
axes[1].set_title("MLP Probe Accuracy by Style × Layer")
axes[1].set_xlabel("Layer")

plt.suptitle("Probing Accuracy Across Layers (Pythia-2.8B)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_layerwise_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: 02_layerwise_heatmap.png")


# ============================================================
# FIGURE 3: Non-linearity index per style
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

nl_indices = [probing[s][str(best_layers[s])]["nonlinearity_index"] for s in STYLES]
colors = ["#c44e52" if nl > 1.01 else "#4c72b0" if nl < 0.99 else "#55a868" for nl in nl_indices]

bars = ax.bar(STYLES, nl_indices, color=colors, alpha=0.85)
ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="NL index = 1.0 (perfectly linear)")
ax.set_ylabel("Non-linearity Index (MLP acc / Linear acc)")
ax.set_xlabel("Style / Emotion")
ax.set_title("Non-linearity Index by Style\n>1.0 = MLP outperforms linear (non-linear encoding), <1.0 = Linear outperforms MLP")
ax.set_xticklabels(STYLES, rotation=45, ha="right")
ax.set_ylim(0.95, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_nonlinearity_index.png", bbox_inches="tight")
plt.close()
print("Saved: 03_nonlinearity_index.png")


# ============================================================
# FIGURE 4: Prompting success across styles
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prompt_scores = [prompting[s]["target_score_mean"] for s in STYLES if s in prompting and "error" not in prompting[s]]
prompt_specs = [prompting[s]["specificity"] for s in STYLES if s in prompting and "error" not in prompting[s]]
prompt_styles = [s for s in STYLES if s in prompting and "error" not in prompting[s]]

# Target scores
ax = axes[0]
bars = ax.bar(prompt_styles, prompt_scores, color="#4C72B0", alpha=0.85)
ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="Success threshold (3/5)")
ax.set_ylabel("Target Style Score (1-5)")
ax.set_xlabel("Style")
ax.set_title("Prompting Success: Style Adherence Score")
ax.set_xticklabels(prompt_styles, rotation=45, ha="right")
ax.set_ylim(1, 5.5)
ax.legend()

# Specificity
ax = axes[1]
colors_spec = ["#55a868" if s > 1.0 else "#c44e52" for s in prompt_specs]
bars = ax.bar(prompt_styles, prompt_specs, color=colors_spec, alpha=0.85)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Specificity (Target - Other score)")
ax.set_xlabel("Style")
ax.set_title("Prompting Specificity: Target vs Other Style")
ax.set_xticklabels(prompt_styles, rotation=45, ha="right")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_prompting_results.png", bbox_inches="tight")
plt.close()
print("Saved: 04_prompting_results.png")


# ============================================================
# FIGURE 5: Linear separability vs Prompting success (scatter)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

shared_styles = [s for s in STYLES if s in prompting and "error" not in prompting[s]]
lin_acc_arr = np.array([probing[s][str(best_layers[s])]["linear_acc"] for s in shared_styles])
prompt_score_arr = np.array([prompting[s]["target_score_mean"] for s in shared_styles])
prompt_spec_arr = np.array([prompting[s]["specificity"] for s in shared_styles])

ax = axes[0]
ax.scatter(lin_acc_arr, prompt_score_arr, s=80, c="#4C72B0", alpha=0.8, zorder=5)
for i, style in enumerate(shared_styles):
    ax.annotate(style, (lin_acc_arr[i], prompt_score_arr[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)
r, p = stats.pearsonr(lin_acc_arr, prompt_score_arr)
ax.set_xlabel("Linear Probe Accuracy")
ax.set_ylabel("Prompting Style Score (1-5)")
ax.set_title(f"Linear Separability vs Prompting Success\nr={r:.3f}, p={p:.3f}")

ax = axes[1]
ax.scatter(lin_acc_arr, prompt_spec_arr, s=80, c="#DD8452", alpha=0.8, zorder=5)
for i, style in enumerate(shared_styles):
    ax.annotate(style, (lin_acc_arr[i], prompt_spec_arr[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)
r, p = stats.pearsonr(lin_acc_arr, prompt_spec_arr)
ax.set_xlabel("Linear Probe Accuracy")
ax.set_ylabel("Prompting Specificity")
ax.set_title(f"Linear Separability vs Prompting Specificity\nr={r:.3f}, p={p:.3f}")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_linearity_vs_prompting.png", bbox_inches="tight")
plt.close()
print("Saved: 05_linearity_vs_prompting.png")


# ============================================================
# FIGURE 6: Steering quality (if available)
# ============================================================
if steering:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steer_styles = [s for s in STYLES if s in steering and "activation_metrics" in steering[s]]

    if steer_styles:
        cohens_d = [steering[s]["activation_metrics"]["cohens_d"] for s in steer_styles]
        sv_acc = [steering[s]["activation_metrics"]["sv_accuracy"] for s in steer_styles]

        ax = axes[0]
        colors_d = ["#55a868" if d > 0.5 else "#c44e52" if d < 0.2 else "#4c72b0" for d in cohens_d]
        ax.bar(steer_styles, cohens_d, color=colors_d, alpha=0.85)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Medium effect (d=0.5)")
        ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.7, label="Small effect (d=0.2)")
        ax.set_ylabel("Cohen's d")
        ax.set_xlabel("Style")
        ax.set_title("Steering Vector Effectiveness (Cohen's d)")
        ax.set_xticklabels(steer_styles, rotation=45, ha="right")
        ax.legend()

        ax = axes[1]
        ax.bar(steer_styles, sv_acc, color="#4C72B0", alpha=0.85)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (0.5)")
        ax.set_ylabel("SV Classification Accuracy")
        ax.set_xlabel("Style")
        ax.set_title("Steering Vector as Classifier")
        ax.set_xticklabels(steer_styles, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/06_steering_effectiveness.png", bbox_inches="tight")
        plt.close()
        print("Saved: 06_steering_effectiveness.png")

        # FIGURE 7: Steering vs Prompting scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        steer_shared = [s for s in steer_styles if s in prompting and "error" not in prompting[s]]
        if len(steer_shared) >= 5:
            d_arr = np.array([steering[s]["activation_metrics"]["cohens_d"] for s in steer_shared])
            p_arr = np.array([prompting[s]["target_score_mean"] for s in steer_shared])

            ax.scatter(d_arr, p_arr, s=100, c="#4C72B0", alpha=0.8, zorder=5)
            for i, style in enumerate(steer_shared):
                ax.annotate(style, (d_arr[i], p_arr[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=10)

            r, p = stats.pearsonr(d_arr, p_arr)
            ax.set_xlabel("Steering Effectiveness (Cohen's d)", fontsize=12)
            ax.set_ylabel("Prompting Style Score (1-5)", fontsize=12)
            ax.set_title(f"Steering Difficulty vs Prompting Difficulty\nPearson r={r:.3f}, p={p:.3f}", fontsize=13)

            # Add trend line
            z = np.polyfit(d_arr, p_arr, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(d_arr.min() - 0.05, d_arr.max() + 0.05, 100)
            ax.plot(x_line, p_line(x_line), "--", color="gray", alpha=0.5)

            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/07_steering_vs_prompting.png", bbox_inches="tight")
            plt.close()
            print("Saved: 07_steering_vs_prompting.png")

        # FIGURE 8: Triple comparison (steering vs probing vs prompting)
        if len(steer_shared) >= 5:
            fig, ax = plt.subplots(figsize=(10, 6))

            lin_acc_s = np.array([probing[s][str(best_layers[s])]["linear_acc"] for s in steer_shared])
            d_arr = np.array([steering[s]["activation_metrics"]["cohens_d"] for s in steer_shared])
            p_arr = np.array([prompting[s]["target_score_mean"] for s in steer_shared])

            scatter = ax.scatter(lin_acc_s, d_arr, s=p_arr * 50, c=p_arr,
                                cmap="RdYlGn", alpha=0.8, edgecolors="black", zorder=5)
            for i, style in enumerate(steer_shared):
                ax.annotate(style, (lin_acc_s[i], d_arr[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=10)

            cbar = plt.colorbar(scatter, label="Prompting Score (1-5)")
            ax.set_xlabel("Linear Probe Accuracy", fontsize=12)
            ax.set_ylabel("Steering Effectiveness (Cohen's d)", fontsize=12)
            ax.set_title("Three Dimensions: Linear Separability × Steering × Prompting\n"
                         "Bubble size = prompting score", fontsize=13)

            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/08_triple_comparison.png", bbox_inches="tight")
            plt.close()
            print("Saved: 08_triple_comparison.png")


# ============================================================
# FIGURE 9: Summary heatmap of all metrics
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

metric_names = ["Linear\nProbe", "MLP\nProbe", "NL\nIndex"]
data_matrix = []
for s in STYLES:
    row = [
        probing[s][str(best_layers[s])]["linear_acc"],
        probing[s][str(best_layers[s])]["mlp_acc"],
        probing[s][str(best_layers[s])]["nonlinearity_index"],
    ]
    if s in prompting and "error" not in prompting[s]:
        row.append(prompting[s]["target_score_mean"] / 5.0)  # normalize to 0-1
        row.append(prompting[s]["specificity"] / 4.0)  # normalize
    else:
        row.extend([np.nan, np.nan])

    if steering and s in steering and "activation_metrics" in steering[s]:
        row.append(min(steering[s]["activation_metrics"]["cohens_d"], 1.0))
        row.append(steering[s]["activation_metrics"]["sv_accuracy"])
    else:
        row.extend([np.nan, np.nan])

    data_matrix.append(row)

if any(not np.isnan(r[3]) for r in data_matrix):
    metric_names.extend(["Prompt\nScore", "Prompt\nSpecific."])
if any(not np.isnan(r[5]) for r in data_matrix):
    metric_names.extend(["Steer\nCohen d", "Steer\nSV Acc"])

data_arr = np.array(data_matrix)[:, :len(metric_names)]
mask = np.isnan(data_arr)

sns.heatmap(data_arr, ax=ax, annot=True, fmt=".3f", mask=mask,
            xticklabels=metric_names, yticklabels=STYLES,
            cmap="YlOrRd", vmin=0.0, vmax=1.0)
ax.set_title("Comprehensive Metrics Overview\nAll values normalized to [0, 1]", fontsize=13)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_comprehensive_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: 09_comprehensive_heatmap.png")


print(f"\nAll plots saved to {PLOTS_DIR}/")
