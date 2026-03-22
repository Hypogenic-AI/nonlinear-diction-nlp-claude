"""
Prompting experiments: Test whether GPT-4.1 can produce each style via prompting.
Then use GPT-4.1 as a judge to evaluate style adherence.
"""

import json
import os
import random
import time
from openai import OpenAI

SEED = 42
random.seed(SEED)

RESULTS_DIR = "/workspaces/nonlinear-diction-nlp-claude/results"

client = OpenAI()  # uses OPENAI_API_KEY

# Styles to test
STYLES = ["joy", "anger", "sadness", "fear", "disgust", "surprise",
          "love", "gratitude", "admiration", "curiosity", "confusion"]

# Prompts to complete in each style
BASE_PROMPTS = [
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
]

STYLE_DESCRIPTIONS = {
    "joy": "joyful and happy",
    "anger": "angry and frustrated",
    "sadness": "sad and melancholic",
    "fear": "fearful and anxious",
    "disgust": "disgusted and repulsed",
    "surprise": "surprised and astonished",
    "love": "loving and affectionate",
    "gratitude": "grateful and thankful",
    "admiration": "admiring and impressed",
    "curiosity": "curious and inquisitive",
    "confusion": "confused and puzzled",
}


def generate_styled_text(style, prompt, model="gpt-4.1-nano"):
    """Generate text in a specific style using prompting."""
    system_msg = (
        f"You are a writer who always writes in a {STYLE_DESCRIPTIONS[style]} tone. "
        f"Continue the given text in that emotional style. Write 1-3 sentences. "
        f"Do not mention the emotion explicitly - just embody it naturally."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Continue this text: \"{prompt}\""},
        ],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].message.content


def judge_style(text, target_style, model="gpt-4.1-nano"):
    """Use GPT-4.1 as judge to rate how well text matches target style."""
    system_msg = (
        "You are an expert judge of emotional tone in text. "
        "Rate how strongly the given text expresses the target emotion on a scale of 1-5:\n"
        "1 = No trace of this emotion\n"
        "2 = Slight hint\n"
        "3 = Moderate\n"
        "4 = Strong\n"
        "5 = Very strong, clearly dominant\n"
        "Respond with ONLY a single number (1-5)."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Target emotion: {target_style}\n\nText: \"{text}\""},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    try:
        score = int(response.choices[0].message.content.strip()[0])
        return min(max(score, 1), 5)
    except (ValueError, IndexError):
        return 3  # default if parsing fails


def judge_style_batch(texts, target_style, model="gpt-4.1-nano"):
    """Judge all texts for a given style, also check if they match OTHER styles (specificity)."""
    scores = []
    for text in texts:
        score = judge_style(text, target_style, model)
        scores.append(score)
        time.sleep(0.1)  # rate limiting
    return scores


def main():
    print("=" * 60)
    print("PROMPTING EXPERIMENT")
    print("=" * 60)

    all_results = {}
    all_generations = {}

    for style in STYLES:
        print(f"\n{'='*50}")
        print(f"Style: {style}")
        print(f"{'='*50}")

        # Generate styled text
        generations = []
        for prompt in BASE_PROMPTS:
            try:
                text = generate_styled_text(style, prompt)
                generations.append({"prompt": prompt, "generation": text})
                print(f"  Prompt: '{prompt}' -> {text[:80]}...")
            except Exception as e:
                print(f"  ERROR: {e}")
                generations.append({"prompt": prompt, "generation": "", "error": str(e)})
            time.sleep(0.2)

        all_generations[style] = generations

        # Judge the generations
        gen_texts = [g["generation"] for g in generations if g["generation"]]

        if gen_texts:
            # How well does it match the TARGET style?
            target_scores = judge_style_batch(gen_texts, style)
            mean_target = sum(target_scores) / len(target_scores)

            # How well does it match a RANDOM other style? (specificity check)
            other_style = random.choice([s for s in STYLES if s != style])
            other_scores = judge_style_batch(gen_texts, other_style)
            mean_other = sum(other_scores) / len(other_scores)

            success_rate = sum(1 for s in target_scores if s >= 3) / len(target_scores)

            all_results[style] = {
                "target_score_mean": round(mean_target, 2),
                "target_scores": target_scores,
                "other_style_tested": other_style,
                "other_score_mean": round(mean_other, 2),
                "other_scores": other_scores,
                "success_rate": round(success_rate, 4),
                "n_generations": len(gen_texts),
                "specificity": round(mean_target - mean_other, 2),
            }

            print(f"\n  Target score ({style}): {mean_target:.2f}/5")
            print(f"  Other score ({other_style}): {mean_other:.2f}/5")
            print(f"  Success rate (≥3): {success_rate:.1%}")
            print(f"  Specificity: {mean_target - mean_other:+.2f}")
        else:
            all_results[style] = {"error": "No generations produced"}

    # Also generate BASELINE (no style instruction) for comparison
    print(f"\n{'='*50}")
    print("Baseline (no style instruction)")
    print(f"{'='*50}")
    baseline_gens = []
    for prompt in BASE_PROMPTS:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "user", "content": f"Continue this text with 1-3 sentences: \"{prompt}\""},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            text = response.choices[0].message.content
            baseline_gens.append(text)
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(0.2)

    # Judge baseline for each style
    baseline_style_scores = {}
    for style in STYLES:
        scores = judge_style_batch(baseline_gens, style)
        baseline_style_scores[style] = {
            "mean": round(sum(scores) / len(scores), 2),
            "scores": scores,
        }
        print(f"  Baseline {style} score: {sum(scores)/len(scores):.2f}/5")

    all_results["baseline_scores"] = baseline_style_scores
    all_generations["baseline"] = baseline_gens

    # Summary
    print("\n" + "=" * 80)
    print("PROMPTING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Style':<15} {'Target':>8} {'Success':>8} {'Specificity':>12}")
    print("-" * 43)
    for style in STYLES:
        if "error" not in all_results[style]:
            r = all_results[style]
            print(f"{style:<15} {r['target_score_mean']:>8.2f} {r['success_rate']:>8.1%} "
                  f"{r['specificity']:>+12.2f}")

    # Save
    with open(f"{RESULTS_DIR}/prompting_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(f"{RESULTS_DIR}/prompting_generations.json", "w") as f:
        json.dump(all_generations, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
