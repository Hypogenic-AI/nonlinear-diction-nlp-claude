"""
Analysis: Correlate steering difficulty, prompting difficulty, and non-linearity index.
"""

import json
import numpy as np
from scipy import stats
import os

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"

# Load all results
with open(f"{RESULTS_DIR}/probing_results.json") as f:
    probing = json.load(f)

with open(f"{RESULTS_DIR}/best_layers.json") as f:
    best_layers = json.load(f)

with open(f"{RESULTS_DIR}/prompting_results.json") as f:
    prompting = json.load(f)

# Load steering if available
steering = None
if os.path.exists(f"{RESULTS_DIR}/steering_results.json"):
    with open(f"{RESULTS_DIR}/steering_results.json") as f:
        steering = json.load(f)

STYLES = [s for s in probing.keys() if s != "neutral"]

print("=" * 80)
print("INTEGRATED ANALYSIS")
print("=" * 80)

# Collect per-style metrics
metrics = {}
for style in STYLES:
    layer = str(best_layers[style])
    p = probing[style][layer]

    m = {
        "linear_acc": p["linear_acc"],
        "mlp_acc": p["mlp_acc"],
        "nonlinearity_index": p["nonlinearity_index"],
        "mlp_minus_linear": p["mlp_minus_linear"],
        "best_layer": int(layer),
    }

    # Prompting metrics
    if style in prompting and "error" not in prompting[style]:
        pm = prompting[style]
        m["prompt_success"] = pm["success_rate"]
        m["prompt_score"] = pm["target_score_mean"]
        m["prompt_specificity"] = pm["specificity"]

    # Steering metrics
    if steering and style in steering:
        s = steering[style]
        if "activation_metrics" in s:
            am = s["activation_metrics"]
            m["steer_cohens_d"] = am["cohens_d"]
            m["steer_sv_accuracy"] = am["sv_accuracy"]
            m["steer_cos_separation"] = am["cos_separation"]
        if "generation_metrics" in s:
            gm = s["generation_metrics"]
            m["steer_rate_shift"] = gm["style_rate_shift"]
            m["steer_prob_shift"] = gm["prob_shift"]

    metrics[style] = m

# Print summary table
print("\n" + "=" * 100)
print("PER-STYLE METRICS SUMMARY")
print("=" * 100)

header = f"{'Style':<13} {'Lin':>6} {'MLP':>6} {'NL_idx':>7} {'Δ':>6} "
if any("steer_cohens_d" in m for m in metrics.values()):
    header += f"{'d':>7} {'SV_acc':>7} "
header += f"{'P_score':>8} {'P_spec':>7}"
print(header)
print("-" * len(header))

for style in STYLES:
    m = metrics[style]
    line = f"{style:<13} {m['linear_acc']:>6.3f} {m['mlp_acc']:>6.3f} {m['nonlinearity_index']:>7.3f} {m['mlp_minus_linear']:>+6.3f} "
    if "steer_cohens_d" in m:
        line += f"{m['steer_cohens_d']:>7.3f} {m['steer_sv_accuracy']:>7.3f} "
    if "prompt_score" in m:
        line += f"{m['prompt_score']:>8.2f} {m.get('prompt_specificity', 0):>+7.2f}"
    print(line)

# Correlation Analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

styles_with_all = [s for s in STYLES if "prompt_score" in metrics[s]]

linear_accs = np.array([metrics[s]["linear_acc"] for s in styles_with_all])
mlp_accs = np.array([metrics[s]["mlp_acc"] for s in styles_with_all])
nl_indices = np.array([metrics[s]["nonlinearity_index"] for s in styles_with_all])
mlp_deltas = np.array([metrics[s]["mlp_minus_linear"] for s in styles_with_all])
prompt_scores = np.array([metrics[s]["prompt_score"] for s in styles_with_all])
prompt_specs = np.array([metrics[s]["prompt_specificity"] for s in styles_with_all])

print("\n1. Linear probe accuracy vs Prompting score:")
r, p = stats.pearsonr(linear_accs, prompt_scores)
rho, p_rho = stats.spearmanr(linear_accs, prompt_scores)
print(f"   Pearson r={r:.3f}, p={p:.4f}")
print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

print("\n2. Linear probe accuracy vs Prompting specificity:")
r, p = stats.pearsonr(linear_accs, prompt_specs)
rho, p_rho = stats.spearmanr(linear_accs, prompt_specs)
print(f"   Pearson r={r:.3f}, p={p:.4f}")
print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

print("\n3. Non-linearity index (MLP/Linear) vs Prompting score:")
r, p = stats.pearsonr(nl_indices, prompt_scores)
rho, p_rho = stats.spearmanr(nl_indices, prompt_scores)
print(f"   Pearson r={r:.3f}, p={p:.4f}")
print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

print("\n4. MLP-Linear delta vs Prompting specificity:")
r, p = stats.pearsonr(mlp_deltas, prompt_specs)
rho, p_rho = stats.spearmanr(mlp_deltas, prompt_specs)
print(f"   Pearson r={r:.3f}, p={p:.4f}")
print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

if any("steer_cohens_d" in metrics[s] for s in styles_with_all):
    styles_with_steer = [s for s in styles_with_all if "steer_cohens_d" in metrics[s]]
    if len(styles_with_steer) >= 5:
        steer_d = np.array([metrics[s]["steer_cohens_d"] for s in styles_with_steer])
        steer_acc = np.array([metrics[s]["steer_sv_accuracy"] for s in styles_with_steer])
        lin_a = np.array([metrics[s]["linear_acc"] for s in styles_with_steer])
        pr_s = np.array([metrics[s]["prompt_score"] for s in styles_with_steer])
        nl_i = np.array([metrics[s]["nonlinearity_index"] for s in styles_with_steer])

        print("\n5. Cohen's d (steering) vs Linear probe accuracy:")
        r, p = stats.pearsonr(steer_d, lin_a)
        rho, p_rho = stats.spearmanr(steer_d, lin_a)
        print(f"   Pearson r={r:.3f}, p={p:.4f}")
        print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

        print("\n6. Cohen's d (steering) vs Prompting score:")
        r, p = stats.pearsonr(steer_d, pr_s)
        rho, p_rho = stats.spearmanr(steer_d, pr_s)
        print(f"   Pearson r={r:.3f}, p={p:.4f}")
        print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

        print("\n7. Cohen's d (steering) vs Non-linearity index:")
        r, p = stats.pearsonr(steer_d, nl_i)
        rho, p_rho = stats.spearmanr(steer_d, nl_i)
        print(f"   Pearson r={r:.3f}, p={p:.4f}")
        print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

        print("\n8. SV accuracy vs Prompting score:")
        r, p = stats.pearsonr(steer_acc, pr_s)
        rho, p_rho = stats.spearmanr(steer_acc, pr_s)
        print(f"   Pearson r={r:.3f}, p={p:.4f}")
        print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

        if any("steer_prob_shift" in metrics[s] for s in styles_with_steer):
            styles_gen = [s for s in styles_with_steer if "steer_prob_shift" in metrics[s]]
            if len(styles_gen) >= 5:
                prob_shift = np.array([metrics[s]["steer_prob_shift"] for s in styles_gen])
                lin_a2 = np.array([metrics[s]["linear_acc"] for s in styles_gen])
                pr_s2 = np.array([metrics[s]["prompt_score"] for s in styles_gen])

                print("\n9. Steering prob shift vs Prompting score:")
                r, p = stats.pearsonr(prob_shift, pr_s2)
                rho, p_rho = stats.spearmanr(prob_shift, pr_s2)
                print(f"   Pearson r={r:.3f}, p={p:.4f}")
                print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

                print("\n10. Steering prob shift vs Linear probe accuracy:")
                r, p = stats.pearsonr(prob_shift, lin_a2)
                rho, p_rho = stats.spearmanr(prob_shift, lin_a2)
                print(f"   Pearson r={r:.3f}, p={p:.4f}")
                print(f"   Spearman ρ={rho:.3f}, p={p_rho:.4f}")

# Key findings
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Check if NL indices are all near 1
nl_range = nl_indices.max() - nl_indices.min()
nl_mean = nl_indices.mean()
print(f"\n1. Non-linearity index range: [{nl_indices.min():.3f}, {nl_indices.max():.3f}], mean={nl_mean:.3f}")
if nl_range < 0.1 and abs(nl_mean - 1.0) < 0.05:
    print("   → All styles are approximately LINEARLY represented (NL index ≈ 1.0)")
    print("   → MLP probes do NOT outperform linear probes for any style")
    print("   → This SUPPORTS the null hypothesis for representation linearity")

# Check if linear accuracy varies
print(f"\n2. Linear probe accuracy range: [{linear_accs.min():.3f}, {linear_accs.max():.3f}]")
print(f"   Hardest to linearly separate: {styles_with_all[np.argmin(linear_accs)]}")
print(f"   Easiest to linearly separate: {styles_with_all[np.argmax(linear_accs)]}")

# Check prompting uniformity
print(f"\n3. Prompting score range: [{prompt_scores.min():.2f}, {prompt_scores.max():.2f}]")
print(f"   All styles achieved ≥{prompt_scores.min():.1f}/5 via prompting")
if prompt_scores.min() >= 3.0:
    print("   → Prompting succeeds for ALL styles, even those with lower linear separability")

# Save comprehensive metrics
with open(f"{RESULTS_DIR}/comprehensive_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nComprehensive metrics saved to {RESULTS_DIR}/comprehensive_metrics.json")
