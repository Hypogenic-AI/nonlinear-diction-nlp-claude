"""
Fast steering experiment: Measure steering effectiveness using activation-space metrics.
Instead of generating + classifying (slow), we:
1. Compute steering vectors
2. Generate steered text
3. Measure style shift using cosine similarity with the steering direction
"""

import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"

with open(f"{RESULTS_DIR}/metadata.json") as f:
    metadata = json.load(f)

with open(f"{RESULTS_DIR}/best_layers.json") as f:
    best_layers = json.load(f)

STYLES = [s for s in metadata["styles"] if s != "neutral"]

print("Loading activations...")
data = np.load(f"{RESULTS_DIR}/activations.npz")


def compute_steering_vector(style, layer):
    """Mean difference vector."""
    style_acts = data[f"{style}_layer{layer}"]
    neutral_acts = data[f"neutral_layer{layer}"]
    return style_acts.mean(axis=0) - neutral_acts.mean(axis=0)


def measure_steering_quality(style, layer):
    """
    Measure how effective a steering vector would be, using activation-space metrics:
    1. Cosine similarity between SV and individual style activations
    2. Classification accuracy using SV as a 1D projection
    3. Effect size (Cohen's d) of the style vs neutral projections
    """
    sv = compute_steering_vector(style, layer)
    sv_norm = np.linalg.norm(sv)
    sv_unit = sv / (sv_norm + 1e-8)

    style_acts = data[f"{style}_layer{layer}"]
    neutral_acts = data[f"neutral_layer{layer}"]

    # Project all activations onto SV direction
    style_proj = style_acts @ sv_unit
    neutral_proj = neutral_acts @ sv_unit

    # Cohen's d effect size
    pooled_std = np.sqrt((style_proj.std()**2 + neutral_proj.std()**2) / 2)
    cohens_d = (style_proj.mean() - neutral_proj.mean()) / (pooled_std + 1e-8)

    # Classification accuracy using SV threshold
    threshold = (style_proj.mean() + neutral_proj.mean()) / 2
    style_correct = (style_proj > threshold).mean()
    neutral_correct = (neutral_proj <= threshold).mean()
    sv_accuracy = (style_correct + neutral_correct) / 2

    # Cosine similarities
    style_cos = np.array([
        np.dot(act, sv_unit) / (np.linalg.norm(act) + 1e-8) for act in style_acts
    ])
    neutral_cos = np.array([
        np.dot(act, sv_unit) / (np.linalg.norm(act) + 1e-8) for act in neutral_acts
    ])

    return {
        "sv_norm": float(sv_norm),
        "cohens_d": float(cohens_d),
        "sv_accuracy": float(sv_accuracy),
        "style_proj_mean": float(style_proj.mean()),
        "neutral_proj_mean": float(neutral_proj.mean()),
        "style_cos_mean": float(style_cos.mean()),
        "neutral_cos_mean": float(neutral_cos.mean()),
        "cos_separation": float(style_cos.mean() - neutral_cos.mean()),
    }


def generate_and_measure_steering(model, tokenizer, style, layer, alpha=3.0, n_prompts=15):
    """Generate with and without steering, measure activation shift."""
    device = model.cfg.device
    sv = compute_steering_vector(style, layer)
    sv_tensor = torch.tensor(sv, dtype=torch.float16, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    test_prompts = [
        "I think that", "The most important thing is", "When I look at the world,",
        "People often say that", "It's interesting how", "I was surprised to find",
        "The truth about life is", "What really matters is", "Sometimes I feel like",
        "In my experience,", "One thing I've noticed is", "Looking back, I realize",
        "The best part about this is", "I can't believe that", "When you think about it,",
    ][:n_prompts]

    def steering_hook(value, hook):
        value[:, :, :] = value + alpha * sv_tensor
        return value

    steered_final_acts = []
    baseline_final_acts = []
    steered_texts = []
    baseline_texts = []

    for prompt in tqdm(test_prompts, desc=f"  Generating ({style}, α={alpha})"):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = tokens["input_ids"].to(device)

        # Baseline generation
        model.reset_hooks()
        with torch.no_grad():
            baseline_out = model.generate(
                input_ids, max_new_tokens=40, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
        baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
        baseline_texts.append(baseline_text)

        # Steered generation
        model.add_hook(hook_name, steering_hook)
        with torch.no_grad():
            steered_out = model.generate(
                input_ids, max_new_tokens=40, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
        model.reset_hooks()
        steered_text = tokenizer.decode(steered_out[0], skip_special_tokens=True)
        steered_texts.append(steered_text)

        # Get final activations for both
        with torch.no_grad():
            b_tokens = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=128)
            _, b_cache = model.run_with_cache(
                b_tokens["input_ids"].to(device), names_filter=[hook_name]
            )
            b_act = b_cache[hook_name][0].mean(dim=0).cpu().numpy()
            baseline_final_acts.append(b_act)
            del b_cache

            s_tokens = tokenizer(steered_text, return_tensors="pt", truncation=True, max_length=128)
            _, s_cache = model.run_with_cache(
                s_tokens["input_ids"].to(device), names_filter=[hook_name]
            )
            s_act = s_cache[hook_name][0].mean(dim=0).cpu().numpy()
            steered_final_acts.append(s_act)
            del s_cache

        torch.cuda.empty_cache()

    # Measure activation shift toward style
    sv_unit = sv / (np.linalg.norm(sv) + 1e-8)
    baseline_proj = np.array([act @ sv_unit for act in baseline_final_acts])
    steered_proj = np.array([act @ sv_unit for act in steered_final_acts])

    activation_shift = float(steered_proj.mean() - baseline_proj.mean())

    # Also use the linear probe from training data as classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    style_acts = data[f"{style}_layer{layer}"]
    neutral_acts = data[f"neutral_layer{layer}"]
    n = min(len(style_acts), len(neutral_acts), 400)
    X_train = np.concatenate([style_acts[:n], neutral_acts[:n]])
    y_train = np.concatenate([np.ones(n), np.zeros(n)])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_s, y_train)

    baseline_preds = clf.predict(scaler.transform(np.array(baseline_final_acts)))
    steered_preds = clf.predict(scaler.transform(np.array(steered_final_acts)))
    baseline_probs = clf.predict_proba(scaler.transform(np.array(baseline_final_acts)))[:, 1]
    steered_probs = clf.predict_proba(scaler.transform(np.array(steered_final_acts)))[:, 1]

    return {
        "activation_shift": round(activation_shift, 4),
        "baseline_style_rate": round(float(baseline_preds.mean()), 4),
        "steered_style_rate": round(float(steered_preds.mean()), 4),
        "style_rate_shift": round(float(steered_preds.mean() - baseline_preds.mean()), 4),
        "baseline_prob_mean": round(float(baseline_probs.mean()), 4),
        "steered_prob_mean": round(float(steered_probs.mean()), 4),
        "prob_shift": round(float(steered_probs.mean() - baseline_probs.mean()), 4),
        "steered_texts": steered_texts[:3],  # save a few examples
        "baseline_texts": baseline_texts[:3],
    }


def main():
    # Phase 1: Activation-space metrics (fast, no model needed)
    print("=" * 60)
    print("Phase 1: Activation-space steering quality metrics")
    print("=" * 60)

    activation_results = {}
    for style in STYLES:
        layer = best_layers[style]
        metrics = measure_steering_quality(style, layer)
        activation_results[style] = {**metrics, "layer": layer}
        print(f"  {style:<15} layer={layer:2d}  Cohen's d={metrics['cohens_d']:.3f}  "
              f"SV_acc={metrics['sv_accuracy']:.3f}  cos_sep={metrics['cos_separation']:.4f}")

    # Phase 2: Actual generation + classification (slower)
    print("\n" + "=" * 60)
    print("Phase 2: Generation-based steering measurement")
    print("=" * 60)

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        "EleutherAI/pythia-2.8b",
        device="cuda:0",
        dtype=torch.float16,
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_results = {}
    for style in STYLES:
        print(f"\n{'='*40}")
        print(f"Style: {style}")
        print(f"{'='*40}")

        layer = best_layers[style]
        gen_result = generate_and_measure_steering(
            model, tokenizer, style, layer, alpha=3.0, n_prompts=15
        )
        generation_results[style] = gen_result

        print(f"  Baseline rate: {gen_result['baseline_style_rate']:.3f}")
        print(f"  Steered rate:  {gen_result['steered_style_rate']:.3f}")
        print(f"  Shift:         {gen_result['style_rate_shift']:+.3f}")
        print(f"  Prob shift:    {gen_result['prob_shift']:+.3f}")

    # Random vector control
    print("\n" + "=" * 60)
    print("Control: Random vector steering")
    print("=" * 60)

    random_results = {}
    for style in STYLES[:3]:
        layer = best_layers[style]
        d_model = metadata["d_model"]
        random_sv = np.random.randn(d_model).astype(np.float32)
        real_sv_norm = np.linalg.norm(compute_steering_vector(style, layer))
        random_sv = random_sv / np.linalg.norm(random_sv) * real_sv_norm

        # Quick test with random vector
        random_sv_unit = random_sv / (np.linalg.norm(random_sv) + 1e-8)
        style_acts = data[f"{style}_layer{layer}"]
        neutral_acts = data[f"neutral_layer{layer}"]
        random_style_proj = (style_acts @ random_sv_unit).mean()
        random_neutral_proj = (neutral_acts @ random_sv_unit).mean()
        random_results[style] = {
            "proj_diff": round(float(random_style_proj - random_neutral_proj), 4),
        }
        print(f"  {style}: random projection diff = {random_style_proj - random_neutral_proj:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("STEERING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Style':<15} {'d':>8} {'SV_acc':>8} {'Base':>8} {'Steered':>8} {'Shift':>8}")
    print("-" * 55)
    for style in STYLES:
        a = activation_results[style]
        g = generation_results[style]
        print(f"{style:<15} {a['cohens_d']:>8.3f} {a['sv_accuracy']:>8.3f} "
              f"{g['baseline_style_rate']:>8.3f} {g['steered_style_rate']:>8.3f} "
              f"{g['style_rate_shift']:>+8.3f}")

    # Save all results
    combined = {}
    for style in STYLES:
        combined[style] = {
            "activation_metrics": activation_results[style],
            "generation_metrics": generation_results[style],
        }

    with open(f"{RESULTS_DIR}/steering_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    with open(f"{RESULTS_DIR}/random_steering_results.json", "w") as f:
        json.dump(random_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/steering_results.json")


if __name__ == "__main__":
    main()
