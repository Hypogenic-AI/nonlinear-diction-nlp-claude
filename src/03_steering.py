"""
Steering vector experiments: Extract CAA vectors and measure steering success.
Uses Pythia-2.8B with TransformerLens.
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

# Load metadata and best layers
with open(f"{RESULTS_DIR}/metadata.json") as f:
    metadata = json.load(f)

with open(f"{RESULTS_DIR}/best_layers.json") as f:
    best_layers = json.load(f)

with open(f"{RESULTS_DIR}/style_texts.json") as f:
    style_texts = json.load(f)

STYLES = [s for s in metadata["styles"] if s != "neutral"]

# Load activations for steering vector computation
print("Loading activations...")
data = np.load(f"{RESULTS_DIR}/activations.npz")


def compute_steering_vector(style, layer, data):
    """Compute CAA steering vector: mean(style) - mean(neutral)."""
    style_acts = data[f"{style}_layer{layer}"]
    neutral_acts = data[f"neutral_layer{layer}"]
    sv = style_acts.mean(axis=0) - neutral_acts.mean(axis=0)
    return sv


def generate_with_steering(model, tokenizer, prompts, steering_vector, layer, alpha=3.0,
                           max_new_tokens=50):
    """Generate text with a steering vector added at the specified layer."""
    device = model.cfg.device
    generations = []
    sv_tensor = torch.tensor(steering_vector, dtype=torch.float16, device=device)

    hook_name = f"blocks.{layer}.hook_resid_post"

    def steering_hook(value, hook):
        value[:, :, :] = value + alpha * sv_tensor
        return value

    for prompt in tqdm(prompts, desc="Generating with steering"):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                fwd_hooks=[(hook_name, steering_hook)],
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generations.append(generated_text)

    return generations


def generate_without_steering(model, tokenizer, prompts, max_new_tokens=50):
    """Generate text without steering (baseline)."""
    device = model.cfg.device
    generations = []

    for prompt in tqdm(prompts, desc="Generating baseline"):
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generations.append(generated_text)

    return generations


def classify_style_with_probes(texts, style, layer, model, tokenizer, probe_weights):
    """Use the linear probe to classify generated texts."""
    from sklearn.preprocessing import StandardScaler

    device = model.cfg.device
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    for text in tqdm(texts, desc="Getting activations for classification"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])

        resid = cache[hook_name]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (resid * mask).sum(dim=1) / mask.sum(dim=1)
        activations.append(pooled.cpu().numpy()[0])

        del cache
        torch.cuda.empty_cache()

    return np.array(activations)


def main():
    import transformer_lens
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("Loading model...")
    model = transformer_lens.HookedTransformer.from_pretrained(
        "EleutherAI/pythia-2.8b",
        device="cuda:0",
        dtype=torch.float16,
    )
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts for generation
    test_prompts = [
        "I think that",
        "The most important thing is",
        "When I look at the world,",
        "People often say that",
        "It's interesting how",
        "I was surprised to find",
        "The truth about life is",
        "What really matters is",
        "Sometimes I feel like",
        "In my experience,",
        "One thing I've noticed is",
        "Looking back, I realize",
        "The best part about this is",
        "I can't believe that",
        "When you think about it,",
        "It makes sense that",
        "Everyone knows that",
        "The problem with this is",
        "I've always believed that",
        "What surprised me was",
    ]

    # Generate baseline (unsteered) text
    print("\nGenerating baseline (unsteered) text...")
    baseline_gens = generate_without_steering(model, tokenizer, test_prompts)

    # For each style: compute steering vector, generate steered text, classify
    steering_results = {}
    all_generations = {"baseline": baseline_gens}

    for style in STYLES:
        print(f"\n{'='*50}")
        print(f"Steering: {style}")
        print(f"{'='*50}")

        layer = best_layers[style]
        print(f"Using layer {layer} (best probing layer)")

        # Compute steering vector
        sv = compute_steering_vector(style, layer, data)
        sv_norm = np.linalg.norm(sv)
        print(f"Steering vector norm: {sv_norm:.2f}")

        # Try multiple alpha values
        alphas = [1.0, 3.0, 5.0]
        best_alpha_results = None
        best_shift = -1

        for alpha in alphas:
            print(f"\n  Alpha={alpha}:")
            steered_gens = generate_with_steering(
                model, tokenizer, test_prompts, sv, layer, alpha=alpha
            )

            # Get activations for both baseline and steered text
            # Then train a quick classifier to measure style shift
            style_acts = data[f"{style}_layer{layer}"]
            neutral_acts = data[f"neutral_layer{layer}"]

            # Train classifier on original data
            n = min(len(style_acts), len(neutral_acts), 400)
            X_train = np.concatenate([style_acts[:n], neutral_acts[:n]])
            y_train = np.concatenate([np.ones(n), np.zeros(n)])
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X_train_s, y_train)

            # Classify steered generations
            steered_acts = classify_style_with_probes(
                steered_gens, style, layer, model, tokenizer, None
            )
            steered_acts_s = scaler.transform(steered_acts)
            steered_probs = clf.predict_proba(steered_acts_s)[:, 1]
            steered_preds = clf.predict(steered_acts_s)

            # Classify baseline generations
            baseline_acts = classify_style_with_probes(
                baseline_gens, style, layer, model, tokenizer, None
            )
            baseline_acts_s = scaler.transform(baseline_acts)
            baseline_probs = clf.predict_proba(baseline_acts_s)[:, 1]
            baseline_preds = clf.predict(baseline_acts_s)

            steered_rate = float(steered_preds.mean())
            baseline_rate = float(baseline_preds.mean())
            shift = steered_rate - baseline_rate

            print(f"    Baseline style rate: {baseline_rate:.3f}")
            print(f"    Steered style rate: {steered_rate:.3f}")
            print(f"    Shift: {shift:+.3f}")

            if shift > best_shift:
                best_shift = shift
                best_alpha_results = {
                    "alpha": alpha,
                    "baseline_rate": round(baseline_rate, 4),
                    "steered_rate": round(steered_rate, 4),
                    "shift": round(shift, 4),
                    "steered_probs_mean": round(float(steered_probs.mean()), 4),
                    "baseline_probs_mean": round(float(baseline_probs.mean()), 4),
                    "sv_norm": round(float(sv_norm), 4),
                }
                all_generations[f"{style}_steered"] = steered_gens

        steering_results[style] = best_alpha_results
        print(f"\n  Best alpha={best_alpha_results['alpha']}, shift={best_alpha_results['shift']:+.3f}")

    # Also compute random vector steering as control
    print("\n" + "=" * 50)
    print("Control: Random vector steering")
    print("=" * 50)

    random_results = {}
    for style in STYLES[:3]:  # Just test a few
        layer = best_layers[style]
        d_model = metadata["d_model"]
        random_sv = np.random.randn(d_model).astype(np.float32)
        random_sv = random_sv / np.linalg.norm(random_sv) * np.linalg.norm(
            compute_steering_vector(style, layer, data)
        )

        steered_gens = generate_with_steering(
            model, tokenizer, test_prompts[:10], random_sv, layer, alpha=3.0
        )

        # Classify
        style_acts = data[f"{style}_layer{layer}"]
        neutral_acts = data[f"neutral_layer{layer}"]
        n = min(len(style_acts), len(neutral_acts), 400)
        X_train = np.concatenate([style_acts[:n], neutral_acts[:n]])
        y_train = np.concatenate([np.ones(n), np.zeros(n)])
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train_s, y_train)

        steered_acts = classify_style_with_probes(
            steered_gens, style, layer, model, tokenizer, None
        )
        steered_acts_s = scaler.transform(steered_acts)
        random_rate = float(clf.predict(steered_acts_s).mean())
        random_results[style] = round(random_rate, 4)
        print(f"  {style}: random vector rate = {random_rate:.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("STEERING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Style':<15} {'Alpha':>5} {'Base':>8} {'Steered':>8} {'Shift':>8}")
    print("-" * 44)
    for style in STYLES:
        r = steering_results[style]
        print(f"{style:<15} {r['alpha']:>5.1f} {r['baseline_rate']:>8.3f} "
              f"{r['steered_rate']:>8.3f} {r['shift']:>+8.3f}")

    # Save results
    with open(f"{RESULTS_DIR}/steering_results.json", "w") as f:
        json.dump(steering_results, f, indent=2)

    with open(f"{RESULTS_DIR}/random_steering_results.json", "w") as f:
        json.dump(random_results, f, indent=2)

    with open(f"{RESULTS_DIR}/generations.json", "w") as f:
        json.dump(all_generations, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
