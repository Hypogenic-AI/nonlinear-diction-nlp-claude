# Literature Review: Non-linear Diction

## Research Area Overview

This literature review synthesizes research on **steering vectors**, the **linear representation hypothesis (LRH)**, and evidence for **non-linear concept representations** in large language models. The central research question is: *if there is a particular style for which it is difficult to obtain a steering vector, is it also difficult to prompt for that style?* More broadly, we investigate whether all styles known to LLMs are represented linearly, or whether some are encoded non-linearly.

The field of activation engineering has rapidly matured since 2022, with steering vectors becoming a standard tool for controlling LLM behavior at inference time. The dominant assumption — that concepts (including styles) are represented as linear directions in activation space — has been formalized as the Linear Representation Hypothesis. However, recent work reveals systematic exceptions and limitations that are directly relevant to our hypothesis.

---

## Key Papers

### 1. The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors:** Park, Choe, Veitch (U. Chicago)
- **Year:** 2023 (ICML 2024)
- **arXiv:** 2311.03658
- **Key Contribution:** Formalizes three distinct interpretations of the LRH (subspace, measurement, intervention) and proves they are unified under a causal inner product framework.
- **Methodology:** Counterfactual pair analysis on LLaMA-2-7B. Tests 27 concepts (morphological, semantic, syntactic, cross-lingual) using unembedding/embedding representations.
- **Key Results:** 26/27 concepts show clear linear structure. **One exception: `thing → part`** (e.g., bus → seats) does not have a linear representation — its alignment histogram is indistinguishable from random.
- **Relevance:** The framework is inherently linear — concepts encoded non-linearly would simply appear as "not represented." The sole exception (`thing → part`) is a relational/compositional concept, suggesting complex semantic relationships resist linear encoding. Style, being complex and compositional, may exhibit similar resistance.

### 2. Steering Language Models With Activation Engineering (ActAdd)
- **Authors:** Turner et al.
- **Year:** 2023
- **arXiv:** 2308.10248
- **Key Contribution:** Introduces Activation Addition (ActAdd) — computing steering vectors from contrastive prompt pairs and adding them to residual stream activations.
- **Methodology:** Single prompt pair → compute activation difference → add with scaling coefficient at target layer. Tested on GPT-2-XL, GPT-J-6B, LLaMA-1-13B, OPT-6.7B, LLaMA-3-8B.
- **Styles/Concepts Tested:** Sentiment, topics (weddings, finance, art), toxicity, anger, conspiracy, factual editing.
- **What Failed:** "Art" topic showed zero steering effect. Compositional rules (goose → AAAAHHHH) produced gibberish. Cross-model transfer was inconsistent. Positive-to-negative sentiment was harder than negative-to-positive.
- **Evidence for Non-linearity:** Partial dimension experiments showed non-monotonic behavior (adding 1,120/1,600 dimensions produced MORE wedding content than all 1,600). Superposition effects observed (unrelated tokens boosted by steering vectors).
- **Relevance:** Demonstrates that steering works well for simple binary concepts but degrades for complex, compositional, or abstract targets — directly relevant to our hypothesis about style.

### 3. Style Vectors for Steering Generative Large Language Models
- **Authors:** Konen, Jentzsch et al. (DLR)
- **Year:** 2024 (EACL Findings)
- **arXiv:** 2402.01618
- **Key Contribution:** Most directly relevant prior work. Tests activation-based and training-based style vectors for sentiment, emotion, and writing style on Alpaca-7B.
- **Styles Tested:** Positive/negative sentiment (Yelp), 6 Ekman emotions (GoEmotions), Shakespearean/modern writing style.
- **What Worked:** Positive sentiment, joy, anger, Shakespearean style (at higher λ).
- **What Failed:**
  - **Disgust:** Lowest AUC (0.51, near random) with trained vectors; "does not seem to be steered"
  - **Surprise:** Very few successfully trained vectors; low AUC (0.38-0.39)
  - **Negative sentiment:** Harder to steer than positive
  - **Factual prompts:** Resistant to all style steering
- **Critical Observation:** "More complex styles, such as multidimensional composed styles, present unique challenges when approached through activation engineering" — explicitly acknowledging limits of linear steering for complex styles.
- **Datasets:** Yelp Review (542K), GoEmotions (58K, 5K used), Shakespeare parallel corpus (18K pairs).
- **Code:** https://github.com/DLR-SC/style-vectors-for-steering-llms

### 4. Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations
- **Authors:** Csordas, Potts, Manning, Geiger (Stanford)
- **Year:** 2024
- **arXiv:** 2408.10920
- **Key Contribution:** Discovers **"onion representations"** — the first clean empirical counterexample to the Strong LRH.
- **Mechanism:** In small GRUs, tokens at different positions are stored as the **same direction** at **different magnitudes** (exponential scaling). Decoding requires autoregressive "peeling" — fundamentally non-linear.
- **Evidence:** Linear probing and DAS completely fail (0-1% accuracy) on small models where onion interventions achieve 83-87%.
- **Implications for Steering:**
  - If a concept is encoded via magnitude rather than direction, **linear steering vectors will be ineffective**
  - SAE-based feature identification likewise assumes linear decomposition
  - The field's exclusive reliance on linear methods creates **systematic blind spots**
  - Compression pressure (many features, small residual stream) drives non-linear solutions — same pressure exists in LLMs
- **Relevance:** Provides theoretical and empirical basis for our hypothesis. If style features face compression pressure, they may be encoded non-linearly, explaining why some styles resist both steering and prompting.

### 5. Steering Llama 2 via Contrastive Activation Addition (CAA)
- **Authors:** Rimsky et al.
- **Year:** 2024 (ACL)
- **arXiv:** 2312.06681
- **Key Contribution:** Scales ActAdd by averaging activation differences across many contrastive examples. Tests on Llama 2 Chat (7B, 13B).
- **Behaviors Tested:** Sycophancy, hallucination, corrigibility, survival instinct, myopia, power-seeking, wealth-seeking.
- **Results:** Significant behavioral changes with minimal capability reduction. Works on top of RLHF.
- **Relevance:** Demonstrates linear steering works for behavioral dimensions — but these are high-level binary concepts. Style/diction is more continuous and nuanced.

### 6. Toy Models of Superposition
- **Authors:** Elhage et al. (Anthropic)
- **Year:** 2022
- **arXiv:** 2209.10652
- **Key Contribution:** Demonstrates that neural networks store more features than dimensions via superposition. Features organize into geometric polytope structures.
- **Key Results:** Phase diagram governs monosemantic vs. polysemantic neurons. Superposition is driven by feature sparsity and importance. Models can compute while in superposition.
- **Relevance:** Superposition means features are not cleanly separable — recovering them requires non-linear filtering. If style features are sparse and low-importance relative to content features, they will be heavily superposed and thus harder to isolate linearly.

### 7. On the Origins of Linear Representations in Large Language Models
- **Authors:** Jiang et al.
- **Year:** 2024
- **arXiv:** 2403.03867
- **Key Contribution:** Provides theoretical explanation for why LLMs develop linear representations — arises from next-token prediction (softmax + cross-entropy) and gradient descent implicit bias.
- **Limitations of Theory:** Assumes binary concepts with clear contrasts. Non-binary, continuous, or compositional concepts (like style) may not satisfy these conditions.
- **Relevance:** If the conditions for linearity (binary concepts, independent log-odds) are not met for style, the theory actually **predicts** non-linear representations for complex styles.

### 8. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
- **Authors:** Wu et al. (Stanford NLP)
- **Year:** 2025
- **arXiv:** 2501.17148
- **Key Contribution:** First large-scale benchmark comparing steering methods. Tests on 500 concepts for Gemma-2-2B/9B.
- **Key Finding:** **Prompting outperforms all representation-based steering methods.** SAEs are not competitive. Diff-in-means best for detection but weak for steering.
- **Relevance:** If prompting outperforms linear steering, it suggests either (a) prompting accesses representations that linear methods cannot, or (b) the model processes prompts through non-linear pathways. This gap between prompting and linear steering is central to our hypothesis.

### 9. Polysemanticity and Capacity in Neural Networks
- **Authors:** Scherlis et al.
- **Year:** 2022
- **arXiv:** 2210.01892
- **Key Contribution:** Introduces "feature capacity" — fractional dimension consumed per feature. Less important/sparser features are allocated less capacity and become polysemantic.
- **Relevance:** Style features, if less important to the training objective than factual content, would be encoded polysemantically — sharing dimensions and resisting clean linear extraction.

### 10. The Geometry of Categorical and Hierarchical Concepts in LLMs
- **Authors:** Park et al.
- **Year:** 2024 (ICLR 2025)
- **arXiv:** 2406.01506
- **Key Contribution:** Extends LRH to categorical concepts (polytopes) and hierarchical concepts (orthogonal subspaces). Tests on 900+ WordNet concepts.
- **Relevance:** Even within the "linear" framework, complex concept types require geometric structures beyond simple directions. Style/diction concepts are inherently multi-dimensional and hierarchical (formal/informal subsumes many sub-dimensions like vocabulary, sentence length, register).

### 11. Word Embeddings Are Steers for Language Models
- **Authors:** Han et al.
- **Year:** 2023
- **arXiv:** 2305.12798
- **Key Contribution:** Shows output embedding transformations (d × d matrix) can steer style. Uses 0.2% of parameters. Transferable and composable.
- **Relevance:** The need for a full d × d matrix (not just a direction vector) implies style control requires more than a single linear direction — consistent with higher-dimensional or non-linear encoding.

### 12. Improving Steering Vectors by Targeting Sparse Autoencoder Features
- **Authors:** Chalnev et al.
- **Year:** 2024
- **arXiv:** 2411.02193
- **Key Contribution:** Uses SAEs to measure causal effects of steering vectors and construct targeted vectors.
- **Key Finding:** Naive linear steering produces unpredictable side effects — a "months" feature causes year digit outputs. Features are entangled in non-linear/compositional ways.
- **Relevance:** Demonstrates that even with SAE-based feature identification, the relationship between features and outputs is more complex than linear.

---

## Common Methodologies

1. **Contrastive Activation Addition (CAA / ActAdd):** Compute steering vector as mean difference of activations for positive vs. negative examples. Add to residual stream at inference. (Used in papers 2, 3, 5)
2. **Training-based steering vectors:** Optimize vectors via gradient descent to produce target outputs from a frozen model. (Used in papers 3, 11)
3. **Linear probing / DAS:** Train linear classifiers or learn rotation matrices to find concept-specific subspaces. (Used in papers 1, 4, 10)
4. **Sparse Autoencoders (SAEs):** Decompose activations into interpretable, sparse features. (Used in papers 6, 8, 12)
5. **Embedding transformation:** Apply learned linear transforms to output embeddings. (Used in paper 11)

## Standard Baselines

- **Prompting:** "Write in X style" — consistently competitive, often outperforms representation methods (AxBench)
- **PPLM (Plug and Play Language Models):** Gradient-based controlled generation
- **FUDGE:** Controlled generation with future discriminators
- **Unsteered model:** Baseline for measuring steering effect
- **Random vectors:** Control for specificity of steering direction

## Evaluation Metrics

| Metric | Purpose | Used In |
|--------|---------|---------|
| Probing AUC | Linear separability of concept | Papers 1, 3 |
| Intervention accuracy | Causal effect of adding steering vector | Papers 1, 4 |
| Classifier scores (VADER, RoBERTa) | Evaluate style of generated text | Paper 3 |
| Perspective API | Toxicity of generated text | Paper 2 |
| Perplexity / fluency | Quality of steered text | Papers 2, 3, 5, 8 |
| Success rate | Proportion with desired style change | Papers 2, 5 |
| KL divergence | Distribution shift from steering | Paper 2 |
| Cosine similarity | Relevance of steered text to target | Paper 2 |

## Datasets in the Literature

| Dataset | Used In | Task |
|---------|---------|------|
| Yelp Reviews | Papers 3, 11 | Sentiment transfer |
| GoEmotions | Paper 3 | Emotion steering |
| Shakespeare parallel corpus | Paper 3 | Writing style transfer |
| RealToxicityPrompts | Paper 2 | Toxicity reduction |
| IMDb | Paper 2 | Sentiment control |
| OpenWebText | Paper 2 | General generation evaluation |
| BATS 3.0 | Paper 1 | Morphological/semantic analogies |
| WordNet | Paper 10 | Categorical/hierarchical concepts |
| ConceptNet (LAMA) | Paper 2 | Knowledge preservation |

## Gaps and Opportunities

1. **No systematic study of which styles resist linear steering.** Papers test a handful of styles; none systematically evaluate linearity across a broad style taxonomy.
2. **No comparison of steering difficulty vs. prompting difficulty.** Our hypothesis specifically links these — if a style can't be steered, can it be prompted? No paper tests this.
3. **Non-linear representations studied only in small synthetic settings.** The onion representation paper (Csordas et al.) uses small GRUs on sequence copying. Whether similar non-linear encodings exist in large transformers for style/diction features is unexplored.
4. **Limited style diversity.** Most papers test sentiment (binary), basic emotions, or Shakespearean style. Complex stylistic dimensions (irony, understatement, stream-of-consciousness, magical realism, noir) are untested.
5. **Compression pressure in transformers.** The theoretical prediction that compression pressure drives non-linear representations (paper 4) has not been tested empirically for style features in LLMs.

## Recommendations for Our Experiment

### Recommended Datasets
1. **GoEmotions** (primary): 27 emotion categories with varying difficulty — disgust/surprise known to be hard to steer
2. **Yelp Polarity** (baseline): Binary sentiment — known to work with linear steering, serves as positive control
3. **Tiny Shakespeare** (style transfer): Archaic writing style — moderate steering difficulty
4. **Blog Authorship Corpus** (novel): Age/gender-correlated style variation — unexplored for steering
5. **WritingPrompts** (diverse styles): Creative writing with diverse implicit styles — could be labeled post-hoc

### Recommended Baselines
1. **CAA / ActAdd** as the standard linear steering baseline
2. **Prompting baseline** ("Write in X style") for comparison
3. **Diff-in-means** (simple, strong on concept detection per AxBench)
4. **Linear probing AUC** as a measure of linear separability

### Recommended Metrics
1. **Probing accuracy** (linear vs. MLP) — tests whether concept is linearly separable
2. **Steering success rate** — proportion of steered outputs classified as target style
3. **Prompting success rate** — same metric but with prompting instead of steering
4. **Correlation between steering and prompting difficulty** — our core hypothesis test
5. **Non-linearity index:** ratio of MLP probe accuracy to linear probe accuracy (> 1 indicates non-linear encoding)

### Methodological Considerations
- Use a model with good TransformerLens/nnsight support (e.g., Gemma-2-2B, Llama-3-8B)
- Test across multiple layers to identify where style information is encoded
- Include both "easy" styles (sentiment) and "hard" styles (disgust, complex writing styles) for comparison
- Measure non-linearity directly: compare linear probes vs. non-linear probes for each style
- The `steering-vectors` pip library provides a clean API for rapid prototyping
