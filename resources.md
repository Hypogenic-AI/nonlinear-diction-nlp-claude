# Resources Catalog

## Summary

This document catalogs all resources gathered for the **Non-linear Diction** research project, investigating whether some writing styles are represented non-linearly in LLMs, making them difficult to steer with linear vectors.

---

## Papers

Total papers downloaded: **22**

| # | Title | Authors | Year | File | Key Relevance |
|---|-------|---------|------|------|---------------|
| 1 | The Linear Representation Hypothesis | Park et al. | 2023 | papers/2311.03658_*.pdf | Foundational LRH formalization; 1/27 concepts fails |
| 2 | Activation Engineering (ActAdd) | Turner et al. | 2023 | papers/2308.10248_*.pdf | Introduces steering vectors; "art" topic fails |
| 3 | Contrastive Activation Addition (CAA) | Rimsky et al. | 2024 | papers/2312.06681_*.pdf | Scales steering to behavioral dimensions |
| 4 | Style Vectors for Steering | Konen et al. | 2024 | papers/2402.01618_*.pdf | **Most directly relevant** — disgust/surprise fail |
| 5 | Toy Models of Superposition | Elhage et al. | 2022 | papers/2209.10652_*.pdf | Superposition theory; polysemanticity |
| 6 | Extracting Latent Steering Vectors | Subramani et al. | 2022 | papers/2205.05124_*.pdf | Pioneering optimization-based steering |
| 7 | Origins of Linear Representations | Jiang et al. | 2024 | papers/2403.03867_*.pdf | Theory: linearity from cross-entropy + gradient descent |
| 8 | Geometry of Categorical Concepts | Park et al. | 2024 | papers/2406.01506_*.pdf | Polytope structures for non-binary concepts |
| 9 | Non-linear Representations in RNNs | Csordas et al. | 2024 | papers/2408.10920_*.pdf | **Key counterexample** — onion representations |
| 10 | Mean-Centring for Activation Steering | — | 2023 | papers/2312.03813_*.pdf | Improved steering technique |
| 11 | AxBench | Wu et al. | 2025 | papers/2501.17148_*.pdf | Prompting outperforms all steering methods |
| 12 | Word Embeddings Are Steers | Han et al. | 2023 | papers/2305.12798_*.pdf | d×d matrix needed for style — not just direction |
| 13 | Inference-Time Intervention (ITI) | Li et al. | 2023 | papers/2306.03341_*.pdf | Probing + intervention for truthfulness |
| 14 | Plug and Play LMs (PPLM) | Dathathri et al. | 2019 | papers/1912.02164_*.pdf | Early controlled generation baseline |
| 15 | Emergent Linear Representations | Nanda et al. | 2023 | papers/2309.00941_*.pdf | Linearity in world models |
| 16 | Polysemanticity and Capacity | Scherlis et al. | 2022 | papers/2210.01892_*.pdf | Feature capacity framework |
| 17 | Geometry of Concepts (SAE) | Li et al. | 2024 | papers/2410.19750_*.pdf | Multi-scale geometric structure of features |
| 18 | Unified Steering Evaluation | — | 2025 | papers/2502.14829_*.pdf | Steering method comparison |
| 19 | Steering Without Side Effects | Stickland et al. | 2024 | papers/2406.15518_*.pdf | Side effects of linear steering |
| 20 | SAE-Targeted Steering | Chalnev et al. | 2024 | papers/2411.02193_*.pdf | SAE features entangled; unexpected side effects |
| 21 | Language Models Represent Space/Time | Gurnee & Tegmark | 2023 | papers/2310.02207_*.pdf | Linear spatial/temporal reps |
| 22 | Personalized Steering via BPO | — | 2024 | papers/2406.00045_*.pdf | Personalized style steering |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: **5**

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| GoEmotions | HuggingFace | 54K samples, 27 emotions | Emotion classification | datasets/go_emotions/ | Downloaded |
| Yelp Polarity | HuggingFace | 598K samples | Binary sentiment | datasets/yelp_polarity/ | Downloaded |
| Tiny Shakespeare | GitHub | 1.1MB text | Writing style | datasets/tiny_shakespeare/ | Downloaded |
| WritingPrompts | HuggingFace | 5K sample | Diverse creative writing | datasets/writing_prompts/ | Sampled |
| Blog Authorship | HuggingFace | 681K posts | Demographic style | datasets/blog_authorship/ | Download instructions only |

See `datasets/README.md` for download instructions and detailed descriptions.

---

## Code Repositories

Total repositories cloned: **6**

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| style-vectors | github.com/DLR-SC/style-vectors-for-steering-llms | Style steering evaluation | code/style-vectors/ |
| steering-vectors-lib | github.com/steering-vectors/steering-vectors | Pip library for steering | code/steering-vectors-lib/ |
| CAA | github.com/nrimsky/CAA | Contrastive Activation Addition | code/caa/ |
| RepE | github.com/andyzoujm/representation-engineering | Representation Engineering | code/repe/ |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability | code/transformerlens/ |
| pyvene | github.com/stanfordnlp/pyvene | Causal interventions | code/pyvene/ |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for 5 search queries covering steering vectors, linear representation hypothesis, activation engineering, non-linear representations, and style transfer
2. Identified 193 highly relevant papers (R≥3), curated to 22 most relevant
3. Searched HuggingFace, GitHub, and academic sources for datasets and code

### Selection Criteria
- **Papers:** Prioritized work on (a) steering vectors for style/behavior, (b) linear representation hypothesis formalization/testing, (c) evidence for non-linear representations, (d) benchmarks comparing steering methods
- **Datasets:** Focused on style-diverse data with varying expected steering difficulty — from easy (sentiment) to hard (complex emotions, writing styles)
- **Code:** Selected libraries that enable rapid prototyping (steering-vectors-lib), deep analysis (TransformerLens), and non-linear interventions (pyvene)

### Challenges Encountered
- Some HuggingFace datasets use legacy script format (blog_authorship_corpus, tiny_shakespeare) — requires `trust_remote_code=True` or alternative download methods
- Paper 2502.14829 may contain a PDF mismatch (wrong paper content)
- No existing dataset specifically designed for measuring steering difficulty across a broad style taxonomy

### Gaps and Workarounds
- **No "hard styles" benchmark:** No dataset exists with styles annotated by expected steering difficulty. This will need to be constructed from GoEmotions + other sources.
- **No direct prompting-vs-steering comparison dataset:** Our experiments will need to evaluate both prompting and steering on the same styles.
- **Non-linear representations only shown in small models:** The onion representation paper uses small GRUs. Whether similar phenomena exist in transformer LLMs remains our core research question.

---

## Recommendations for Experiment Design

Based on gathered resources:

### 1. Primary Dataset(s)
- **GoEmotions** as primary: 27 emotions provide a natural spectrum of difficulty. Disgust and surprise are known-hard targets.
- **Yelp Polarity** as positive control: Binary sentiment is known to work well with linear steering.
- **Custom style prompts**: Create contrastive prompt pairs for diverse writing styles (formal/informal, verbose/terse, emotional/clinical, literary genres).

### 2. Baseline Methods
- **CAA (Contrastive Activation Addition)** — standard linear steering
- **Prompting** ("Write in X style") — for comparison with steering
- **Diff-in-means** — simplest linear method, strong on detection
- **Linear probing** — measure linear separability as proxy for linearity

### 3. Evaluation Metrics
- **Linear probe accuracy** vs. **MLP probe accuracy** — ratio indicates non-linearity
- **Steering success rate** (classifier-evaluated)
- **Prompting success rate** (same classifier)
- **Correlation: steering difficulty ↔ prompting difficulty** — core hypothesis test
- **Layer-wise probing** to identify where style information is encoded

### 4. Code to Adapt/Reuse
- **steering-vectors-lib** for rapid prototyping of steering experiments
- **TransformerLens** for mechanistic analysis of activation structure
- **pyvene** for non-linear intervention experiments (DAS, causal abstraction)
- **style-vectors** for reference implementation and evaluation pipeline

### 5. Suggested Model
- **Gemma-2-2B** or **Llama-3-8B**: Both well-supported by TransformerLens and the steering-vectors library. Gemma-2-2B recommended for faster iteration; Llama-3-8B for final experiments.
