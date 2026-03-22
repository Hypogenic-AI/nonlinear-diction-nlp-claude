"""
Collect activations from Gemma-2-2B for GoEmotions texts.
For each text, we cache the residual stream activations at selected layers.
"""

import os
import json
import random
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# GoEmotions label map (from the dataset)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# Styles we want to study (subset with known difficulty spectrum)
TARGET_STYLES = {
    "joy": 17,
    "anger": 2,
    "sadness": 25,
    "fear": 14,
    "disgust": 11,
    "surprise": 26,
    "love": 18,
    "gratitude": 15,
    "admiration": 0,
    "curiosity": 7,
    "confusion": 6,
    "neutral": 27,
}

# Layers to probe (Pythia-2.8B has 32 layers)
PROBE_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

MIN_SAMPLES_PER_STYLE = 150
MAX_SAMPLES_PER_STYLE = 500

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_goemotions():
    """Load GoEmotions from local disk."""
    ds = load_from_disk("/workspaces/nonlinear-diction-nlp-claude/datasets/go_emotions/data")
    return ds


def get_style_texts(ds, style_name, style_id, max_n=MAX_SAMPLES_PER_STYLE):
    """Get texts labeled with a specific style (single-label only for clean signal)."""
    texts = []
    for split in ["train", "validation"]:
        for example in ds[split]:
            if example["labels"] == [style_id]:
                texts.append(example["text"])
                if len(texts) >= max_n:
                    return texts
    return texts


def collect_activations_for_texts(model, tokenizer, texts, layers, batch_size=16, max_len=128):
    """
    Run texts through model, collect mean-pooled residual stream activations at specified layers.
    Returns dict: {layer_idx: np.array of shape (n_texts, d_model)}
    """
    device = model.cfg.device
    activations = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=[f"blocks.{l}.hook_resid_post" for l in layers],
            )

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            # Mean pool over non-padding tokens
            resid = cache[hook_name]  # (batch, seq, d_model)
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = (resid * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, d_model)
            activations[layer].append(pooled.cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    for layer in layers:
        activations[layer] = np.concatenate(activations[layer], axis=0)

    return activations


def main():
    print("=" * 60)
    print("Step 1: Loading GoEmotions dataset")
    print("=" * 60)
    ds = load_goemotions()

    # Collect texts per style
    style_texts = {}
    for style_name, style_id in TARGET_STYLES.items():
        texts = get_style_texts(ds, style_name, style_id)
        if len(texts) >= MIN_SAMPLES_PER_STYLE:
            style_texts[style_name] = texts
            print(f"  {style_name}: {len(texts)} texts")
        else:
            print(f"  {style_name}: only {len(texts)} texts (SKIPPING, need {MIN_SAMPLES_PER_STYLE})")

    print(f"\nStyles with enough data: {list(style_texts.keys())}")

    # Save text counts
    text_counts = {k: len(v) for k, v in style_texts.items()}
    with open(f"{RESULTS_DIR}/style_text_counts.json", "w") as f:
        json.dump(text_counts, f, indent=2)

    print("\n" + "=" * 60)
    print("Step 2: Loading Gemma-2-2B with TransformerLens")
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

    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    print("\n" + "=" * 60)
    print("Step 3: Collecting activations for each style")
    print("=" * 60)

    all_activations = {}  # {style_name: {layer: np.array}}
    for style_name, texts in style_texts.items():
        print(f"\nProcessing {style_name} ({len(texts)} texts)...")
        acts = collect_activations_for_texts(model, tokenizer, texts, PROBE_LAYERS, batch_size=32)
        all_activations[style_name] = acts

    # Also collect neutral activations for steering vector computation
    # (neutral = texts not labeled with any emotion)
    neutral_texts = style_texts.get("neutral", [])
    if "neutral" not in all_activations and neutral_texts:
        print(f"\nProcessing neutral ({len(neutral_texts)} texts)...")
        all_activations["neutral"] = collect_activations_for_texts(
            model, tokenizer, neutral_texts, PROBE_LAYERS, batch_size=32
        )

    # Save activations
    print("\nSaving activations...")
    save_path = f"{RESULTS_DIR}/activations.npz"
    save_dict = {}
    for style_name, layer_acts in all_activations.items():
        for layer, arr in layer_acts.items():
            save_dict[f"{style_name}_layer{layer}"] = arr
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved to {save_path} ({os.path.getsize(save_path) / 1e6:.1f} MB)")

    # Save style texts for later use
    with open(f"{RESULTS_DIR}/style_texts.json", "w") as f:
        json.dump(style_texts, f)

    # Save metadata
    metadata = {
        "model": "EleutherAI/pythia-2.8b",
        "probe_layers": PROBE_LAYERS,
        "styles": list(style_texts.keys()),
        "n_texts_per_style": text_counts,
        "max_len": 128,
        "seed": SEED,
        "d_model": model.cfg.d_model,
        "n_layers": model.cfg.n_layers,
    }
    with open(f"{RESULTS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone! Activations collected and saved.")


if __name__ == "__main__":
    main()
