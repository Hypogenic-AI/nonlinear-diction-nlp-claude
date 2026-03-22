"""
Train linear and MLP probes for each style at each layer.
Compute non-linearity index = MLP_acc / linear_acc.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"

# Load metadata
with open(f"{RESULTS_DIR}/metadata.json") as f:
    metadata = json.load(f)

PROBE_LAYERS = metadata["probe_layers"]
STYLES = [s for s in metadata["styles"] if s != "neutral"]
print(f"Styles to probe: {STYLES}")
print(f"Layers: {PROBE_LAYERS}")

# Load activations
print("Loading activations...")
data = np.load(f"{RESULTS_DIR}/activations.npz")

# For each style, binary classification: style vs neutral
neutral_acts = {}
for layer in PROBE_LAYERS:
    neutral_acts[layer] = data[f"neutral_layer{layer}"]

results = {}

for style in STYLES:
    results[style] = {}
    print(f"\n{'='*50}")
    print(f"Probing: {style}")
    print(f"{'='*50}")

    for layer in PROBE_LAYERS:
        style_key = f"{style}_layer{layer}"
        style_acts = data[style_key]
        neut_acts = neutral_acts[layer]

        # Balance classes
        n = min(len(style_acts), len(neut_acts))
        idx_style = np.random.permutation(len(style_acts))[:n]
        idx_neut = np.random.permutation(len(neut_acts))[:n]

        X = np.concatenate([style_acts[idx_style], neut_acts[idx_neut]], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)])

        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        linear_accs = []
        mlp_accs = []
        linear_aucs = []
        mlp_aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Linear probe
            lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
            lr.fit(X_train_s, y_train)
            linear_pred = lr.predict(X_test_s)
            linear_prob = lr.predict_proba(X_test_s)[:, 1]
            linear_accs.append(accuracy_score(y_test, linear_pred))
            linear_aucs.append(roc_auc_score(y_test, linear_prob))

            # MLP probe (2 hidden layers)
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.15,
                learning_rate_init=0.001,
            )
            mlp.fit(X_train_s, y_train)
            mlp_pred = mlp.predict(X_test_s)
            mlp_prob = mlp.predict_proba(X_test_s)[:, 1]
            mlp_accs.append(accuracy_score(y_test, mlp_pred))
            mlp_aucs.append(roc_auc_score(y_test, mlp_prob))

        linear_acc = np.mean(linear_accs)
        mlp_acc = np.mean(mlp_accs)
        linear_auc = np.mean(linear_aucs)
        mlp_auc = np.mean(mlp_aucs)

        # Non-linearity index: how much better MLP is than linear
        # Values > 1.0 indicate non-linear encoding
        nonlinearity_idx = mlp_acc / max(linear_acc, 0.501)  # avoid div by ~0.5

        results[style][str(layer)] = {
            "linear_acc": round(float(linear_acc), 4),
            "linear_acc_std": round(float(np.std(linear_accs)), 4),
            "mlp_acc": round(float(mlp_acc), 4),
            "mlp_acc_std": round(float(np.std(mlp_accs)), 4),
            "linear_auc": round(float(linear_auc), 4),
            "mlp_auc": round(float(mlp_auc), 4),
            "nonlinearity_index": round(float(nonlinearity_idx), 4),
            "mlp_minus_linear": round(float(mlp_acc - linear_acc), 4),
        }

        print(f"  Layer {layer:2d}: Linear={linear_acc:.3f}±{np.std(linear_accs):.3f}  "
              f"MLP={mlp_acc:.3f}±{np.std(mlp_accs):.3f}  "
              f"NL_idx={nonlinearity_idx:.3f}  Δ={mlp_acc-linear_acc:+.3f}")

# Find best layer per style (by linear probe acc)
best_layers = {}
for style in STYLES:
    best_layer = max(PROBE_LAYERS, key=lambda l: results[style][str(l)]["linear_acc"])
    best_layers[style] = best_layer

# Summary table
print("\n" + "=" * 80)
print("SUMMARY: Best layer per style")
print("=" * 80)
print(f"{'Style':<15} {'Layer':>5} {'Linear':>8} {'MLP':>8} {'NL_idx':>8} {'Δ':>8}")
print("-" * 52)
for style in STYLES:
    layer = best_layers[style]
    r = results[style][str(layer)]
    print(f"{style:<15} {layer:>5} {r['linear_acc']:>8.3f} {r['mlp_acc']:>8.3f} "
          f"{r['nonlinearity_index']:>8.3f} {r['mlp_minus_linear']:>+8.3f}")

# Save results
with open(f"{RESULTS_DIR}/probing_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{RESULTS_DIR}/best_layers.json", "w") as f:
    json.dump(best_layers, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}/probing_results.json")
