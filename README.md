# Non-linear Diction: Are Style Representations Linear in LLMs?

## Overview
This project investigates whether language models encode all writing styles linearly, and whether styles that are hard to steer with linear activation vectors are also hard to elicit via prompting. We test 11 emotional styles in Pythia-2.8B using probing, steering vectors (CAA), and prompting (GPT-4.1-nano).

## Key Findings
- **All 11 tested styles are linearly represented** in Pythia-2.8B (non-linearity index ≈ 1.0 for all). MLP probes do not outperform linear probes.
- **Steering effectiveness is independent of linear separability** (r = −0.22, p = 0.52). Some highly linearly-separable styles (love: 0.939 probe acc) steer poorly (Cohen's d = 0.147), and vice versa.
- **Prompting succeeds for ALL styles** (100% success rate, 3.8–4.4/5 scores), including those that steer poorly (disgust: d = 0.129, prompt = 4.4/5).
- **Steering difficulty and prompting difficulty are uncorrelated** (r = −0.018, p = 0.96). This answers the core question: no, hard-to-steer does NOT mean hard-to-prompt.
- The **detection-control gap** is real: a style can be linearly detectable but not linearly controllable. Prompting accesses style representations through nonlinear computational pathways.

## Project Structure
```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and experimental design
├── literature_review.md       # Synthesized literature review
├── resources.md               # Catalog of available resources
├── src/
│   ├── 01_collect_activations.py  # Collect model activations for styles
│   ├── 02_probing.py              # Linear vs MLP probing experiments
│   ├── 03_steering_fast.py        # Steering vector experiments
│   ├── 04_prompting.py            # Prompting experiments (GPT-4.1-nano)
│   ├── 05_analysis.py             # Statistical analysis and correlations
│   └── 06_visualizations.py       # Generate all plots
├── results/
│   ├── probing_results.json       # Probe accuracies per style × layer
│   ├── steering_results.json      # Steering effectiveness metrics
│   ├── prompting_results.json     # Prompting success scores
│   ├── comprehensive_metrics.json # Combined metrics for all styles
│   ├── activations.npz            # Cached model activations (412 MB)
│   └── plots/                     # All visualizations (9 figures)
├── papers/                    # 22 downloaded research papers
├── datasets/                  # GoEmotions, Yelp, Shakespeare, etc.
└── code/                      # Cloned reference repositories
```

## Reproduce
```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch numpy scikit-learn matplotlib scipy pandas datasets transformers accelerate transformer-lens openai seaborn

# Run experiments (in order)
python src/01_collect_activations.py  # ~3 min, needs GPU
python src/02_probing.py              # ~10 min, CPU
python src/03_steering_fast.py        # ~15 min, needs GPU
python src/04_prompting.py            # ~5 min, needs OPENAI_API_KEY
python src/05_analysis.py             # <1 min
python src/06_visualizations.py       # <1 min
```

**Requirements:** NVIDIA GPU (tested on RTX A6000), Python 3.10+, OpenAI API key for prompting experiments.

## See Also
- [REPORT.md](REPORT.md) for full methodology, results, and analysis
- [planning.md](planning.md) for experimental design rationale
