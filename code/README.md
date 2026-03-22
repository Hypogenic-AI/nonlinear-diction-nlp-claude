# Cloned Repositories

## 1. style-vectors (DLR-SC)
- **URL:** https://github.com/DLR-SC/style-vectors-for-steering-llms
- **Location:** `code/style-vectors/`
- **Purpose:** Official implementation of "Style Vectors for Steering Generative LLMs" (EACL 2024). Most directly relevant prior work.
- **Key Files:**
  - Notebooks for training and evaluating style vectors
  - Activation-based and training-based vector computation
  - Evaluation on GoEmotions, Yelp, Shakespeare
- **How to Use:** Follow README for setup. Requires Alpaca-7B / LLaMA-7B.
- **Relevance:** Direct baseline for our experiments. Their code can be adapted to test additional styles.

## 2. steering-vectors-lib
- **URL:** https://github.com/steering-vectors/steering-vectors
- **Location:** `code/steering-vectors-lib/`
- **Purpose:** Pip-installable Python library (`pip install steering-vectors`) for training and applying steering vectors to HuggingFace language models.
- **Key Files:**
  - `steering_vectors/` - core library
  - Supports GPT, LLaMA, Mistral, Gemma models
- **How to Use:** `pip install steering-vectors`, then use API to compute and apply vectors.
- **Relevance:** Best library for rapid prototyping of steering experiments. Clean API for compute → apply → evaluate workflow.

## 3. CAA (Contrastive Activation Addition)
- **URL:** https://github.com/nrimsky/CAA
- **Location:** `code/caa/`
- **Purpose:** Official code for "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024).
- **Key Files:**
  - Scripts for computing contrastive vectors
  - Evaluation on behavioral dimensions (sycophancy, hallucination, etc.)
  - Pre-computed steering vectors included
- **How to Use:** Requires Llama 2 Chat models. Scripts for vector computation and evaluation.
- **Relevance:** Standard CAA implementation. Can be adapted for style-based contrastive pairs.

## 4. RepE (Representation Engineering)
- **URL:** https://github.com/andyzoujm/representation-engineering
- **Location:** `code/repe/`
- **Purpose:** Official repo for "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., 2023). RepReading and RepControl pipelines.
- **Key Files:**
  - RepE library built on HuggingFace
  - Examples for truthfulness, safety, power-seeking
- **How to Use:** `pip install -e .` then use RepReading for probing or RepControl for steering.
- **Relevance:** Alternative framework to CAA. RepControl provides another linear steering method for comparison.

## 5. TransformerLens
- **URL:** https://github.com/TransformerLensOrg/TransformerLens
- **Location:** `code/transformerlens/`
- **Purpose:** Standard library for mechanistic interpretability. HookPoints on every activation, caching, patching, direct logit attribution.
- **Key Files:**
  - `transformer_lens/` - core library
  - Supports 50+ models
- **How to Use:** `pip install transformer-lens`. Load model, cache activations, apply hooks.
- **Relevance:** Essential for mechanistic analysis of *why* certain styles are non-linearly represented. Best for understanding internal representations.

## 6. pyvene (Stanford NLP)
- **URL:** https://github.com/stanfordnlp/pyvene
- **Location:** `code/pyvene/`
- **Purpose:** Library for causal interventions on PyTorch models. Supports DAS (Distributed Alignment Search), interchange interventions, causal abstraction.
- **Key Files:**
  - `pyvene/` - core library
  - Tutorials and examples
- **How to Use:** `pip install pyvene`. Supports any PyTorch model.
- **Relevance:** For testing non-linear interventions (e.g., the onion intervention from Csordas et al.) and comparing linear vs. non-linear probing/intervention methods.
