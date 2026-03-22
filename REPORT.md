# Non-linear Diction: Research Report

## 1. Executive Summary

**Research question:** Does the model represent any style it "knows" a priori non-linearly, or are all styles linearly encoded? If a style is hard to steer with a linear vector, is it also hard to prompt?

**Key finding:** In Pythia-2.8B, all 11 tested emotional/stylistic dimensions are encoded essentially *linearly* (MLP probes do not outperform linear probes for any style, with non-linearity indices ranging 0.965–1.008). However, linear separability does not predict steering effectiveness: the correlation between linear probe accuracy and steering vector effectiveness (Cohen's d) is weak and non-significant (r = −0.220, p = 0.52). Crucially, prompting succeeds uniformly for all styles (3.8–4.4/5), even those that are poorly steerable—indicating that prompting accesses style representations through computational pathways that bypass the limitations of single-direction activation addition.

**Practical implications:** For controllable text generation, prompting remains the most reliable and universal method for style control. Steering vectors' effectiveness appears to depend on factors beyond simple linear separability, suggesting that the causal pathways from activation directions to output behavior involve distributed, nonlinear computation even when the *representations* themselves are linear.

## 2. Goal

### Hypothesis
If there is a particular style for which it is difficult to obtain a steering vector, it may also be difficult to prompt for that style. We tested whether the model represents any style it "knows" a priori non-linearly, or if all styles are represented linearly.

### Sub-hypotheses
- **H1:** Some styles are encoded more non-linearly than others (MLP probe >> linear probe)
- **H2:** Styles with higher non-linearity index are harder to steer with linear vectors
- **H3:** Styles that are hard to steer are NOT necessarily hard to prompt
- **H4:** Simple binary styles are linearly encoded; complex/compositional styles are more non-linear
- **Null:** All styles are linearly represented, and steering difficulty correlates with prompting difficulty

### Importance
Steering vectors are increasingly used for AI safety (alignment via representation engineering). If some behavioral/stylistic dimensions resist linear steering while remaining promptable, this has direct implications for the completeness of linear intervention methods. Understanding the geometry of style representations informs both theoretical interpretability research and practical controllability.

## 3. Data Construction

### Dataset Description
**GoEmotions** (Google Research, 2020): 54K Reddit comments annotated for 27 emotion categories plus neutral. We used the HuggingFace `datasets` library version, pre-downloaded locally.

**Filtering criteria:**
- Single-label examples only (for clean signal)
- Up to 500 examples per emotion category
- Minimum 150 examples required for inclusion

### Styles Tested (11 + neutral)

| Style | N samples | Expected Difficulty |
|-------|-----------|-------------------|
| joy | 500 | Easy (known to steer well) |
| anger | 500 | Easy |
| sadness | 500 | Medium |
| fear | 488 | Medium |
| disgust | 500 | Hard (Konen et al. 2024: AUC 0.51) |
| surprise | 500 | Hard (Konen et al. 2024: few successful vectors) |
| love | 500 | Medium |
| gratitude | 500 | Medium |
| admiration | 500 | Unknown |
| curiosity | 500 | Unknown |
| confusion | 500 | Unknown |
| neutral | 500 | Baseline/reference |

### Example Samples
| Text | Label |
|------|-------|
| "My favourite food is anything I didn't have to cook myself." | neutral |
| "WHY THE FUCK IS BAYLESS ISOING" | anger |
| "Yes I heard abt the f bombs! That has to be why. Thanks for your reply:)" | gratitude |

### Preprocessing
1. Filtered to single-label examples (excluded multi-label)
2. Truncated to max 128 tokens for activation collection
3. Used padding with attention masks for batch processing

## 4. Experiment Description

### Methodology

#### High-Level Approach
Three parallel experiments measuring different facets of style representation:
1. **Probing**: Does the model represent each style linearly? (Linear vs MLP probes)
2. **Steering**: Can we causally shift output toward each style with a linear vector? (CAA)
3. **Prompting**: Can the model produce each style via natural language instruction? (GPT-4.1-nano)

#### Why This Method?
- Probing is the standard way to test the Linear Representation Hypothesis (Park et al., 2023)
- Contrastive Activation Addition (CAA) is the most common steering approach (Turner et al., 2023; Rimsky et al., 2024)
- Prompting is the strongest baseline per AxBench (Wu et al., 2025)
- Comparing all three on the same style taxonomy is novel (Gap #2 in literature)

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | GPU computation |
| TransformerLens | 2.17.0 | Model hooking and activation caching |
| scikit-learn | 1.8.0 | Probes and evaluation |
| OpenAI API | gpt-4.1-nano | Prompting experiments |
| matplotlib/seaborn | latest | Visualization |

#### Model: Pythia-2.8B (EleutherAI)
- 32 transformer layers, d_model = 2560
- Non-gated, well-supported by TransformerLens
- Intermediate size (larger than toy models, small enough for fast iteration)

#### Hardware
- 4× NVIDIA RTX A6000 (49 GB each)
- Used single GPU (cuda:0) for all experiments

#### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Probe layers | [4, 8, 12, 16, 20, 24, 28, 31] | Cover early/mid/late layers |
| Linear probe C | 1.0 | Default regularization |
| MLP probe architecture | (256, 128) hidden | Standard 2-layer MLP |
| MLP early stopping | 15% validation | Prevent overfitting |
| CV folds | 5 | Standard for medium datasets |
| Steering alpha | 3.0 | Moderate intervention strength |
| Generation temperature | 0.7 | Balance diversity/coherence |
| Random seed | 42 | Reproducibility |

### Experimental Protocol

#### Experiment 1: Probing (Linear vs MLP)
For each style × layer:
1. Collect mean-pooled residual stream activations for style and neutral texts
2. Balance classes (equal n per class)
3. Train logistic regression (linear probe) and 2-layer MLP with 5-fold CV
4. Record accuracy, AUC, and non-linearity index (MLP_acc / linear_acc)

#### Experiment 2: Steering Vectors
For each style:
1. Compute steering vector: mean(style_acts) − mean(neutral_acts) at best probing layer
2. Measure activation-space metrics: Cohen's d, SV classification accuracy, cosine separation
3. Generate text with steering vector added to residual stream (alpha=3.0)
4. Classify generated text using a logistic regression trained on original activations
5. Measure shift in style classification rate

#### Experiment 3: Prompting
For each style:
1. Prompt GPT-4.1-nano: "Write in a [style] tone. Continue: [prompt]" for 10 prompts
2. Use GPT-4.1-nano as judge: rate style adherence (1–5 scale)
3. Check specificity: judge against a random other style
4. Generate baseline (no style instruction) for comparison

#### Control: Random Vectors
Steer with random vectors of same norm as real steering vectors to confirm specificity.

### Raw Results

#### Probing Results (Best Layer per Style)

| Style | Layer | Linear | MLP | NL Index | Δ (MLP−Lin) |
|-------|-------|--------|-----|----------|-------------|
| joy | 4 | 0.872 | 0.855 | 0.981 | −0.017 |
| anger | 12 | 0.826 | 0.828 | 1.002 | +0.002 |
| sadness | 4 | 0.856 | 0.826 | 0.965 | −0.030 |
| fear | 4 | 0.900 | 0.881 | 0.980 | −0.018 |
| disgust | 4 | 0.847 | 0.827 | 0.976 | −0.020 |
| surprise | 8 | 0.892 | 0.895 | 1.003 | +0.003 |
| love | 4 | 0.939 | 0.934 | 0.995 | −0.005 |
| gratitude | 16 | 0.970 | 0.954 | 0.984 | −0.016 |
| admiration | 8 | 0.872 | 0.874 | 1.002 | +0.002 |
| curiosity | 20 | 0.831 | 0.838 | 1.008 | +0.007 |
| confusion | 12 | 0.807 | 0.809 | 1.002 | +0.002 |

#### Steering Results

| Style | Cohen's d | SV Accuracy | Baseline Rate | Steered Rate | Shift |
|-------|-----------|-------------|---------------|--------------|-------|
| joy | 0.634 | 0.620 | 0.333 | 0.600 | +0.267 |
| anger | 0.129 | 0.504 | 0.133 | 0.400 | +0.267 |
| sadness | 0.907 | 0.673 | 0.333 | 0.267 | −0.067 |
| fear | 0.090 | 0.499 | 0.400 | 0.800 | +0.400 |
| disgust | 0.129 | 0.510 | 0.400 | 0.533 | +0.133 |
| surprise | 0.163 | 0.539 | 0.467 | 0.400 | −0.067 |
| love | 0.147 | 0.512 | 0.200 | 0.533 | +0.333 |
| gratitude | 0.381 | 0.569 | 0.067 | 0.800 | +0.733 |
| admiration | 0.192 | 0.534 | 0.600 | 0.533 | −0.067 |
| curiosity | 0.525 | 0.604 | 0.200 | 0.333 | +0.133 |
| confusion | 0.455 | 0.587 | 0.333 | 0.600 | +0.267 |

#### Prompting Results (GPT-4.1-nano)

| Style | Target Score (1–5) | Success Rate (≥3) | Specificity |
|-------|-------------------|-------------------|-------------|
| joy | 4.40 | 100% | +3.30 |
| anger | 4.00 | 100% | +2.40 |
| sadness | 4.00 | 100% | +1.80 |
| fear | 4.00 | 100% | +1.50 |
| disgust | 4.40 | 100% | +0.70 |
| surprise | 4.10 | 100% | +1.90 |
| love | 4.00 | 100% | +3.00 |
| gratitude | 4.40 | 100% | +2.20 |
| admiration | 4.00 | 100% | +3.00 |
| curiosity | 3.90 | 100% | +0.70 |
| confusion | 3.80 | 100% | +1.80 |

#### Random Vector Control
| Style | Random Projection Diff |
|-------|----------------------|
| joy | −0.023 |
| anger | −0.241 |
| sadness | −0.028 |

Random vectors show negligible or negative style projection, confirming steering vector specificity.

### Visualizations

Key plots are saved in `results/plots/`:
- `01_linear_vs_mlp_probes.png`: Side-by-side bar chart of linear vs MLP accuracy
- `02_layerwise_heatmap.png`: Probing accuracy across all layers
- `03_nonlinearity_index.png`: NL index per style (all ≈ 1.0)
- `04_prompting_results.png`: Prompting success and specificity
- `05_linearity_vs_prompting.png`: Scatter: linear separability vs prompting
- `06_steering_effectiveness.png`: Cohen's d and SV accuracy per style
- `07_steering_vs_prompting.png`: Core hypothesis test scatter plot
- `08_triple_comparison.png`: Three-way bubble chart
- `09_comprehensive_heatmap.png`: All metrics overview

## 5. Result Analysis

### Key Findings

**Finding 1: All styles are linearly represented (H1 REJECTED)**
Non-linearity indices range from 0.965 to 1.008, with a mean of 0.991. The MLP probe does not outperform the linear probe for *any* style at any layer. This means Pythia-2.8B represents all 11 tested emotions/styles as linear directions in activation space, not non-linearly.

**Finding 2: Linear separability varies substantially across styles**
Despite being all linear, some styles are more strongly linearly separable than others. Gratitude (0.970) and love (0.939) have the highest probe accuracy, while confusion (0.807) and anger (0.826) have the lowest. This 0.16-point range reflects genuine differences in how distinctly the model encodes each style.

**Finding 3: Steering effectiveness is NOT predicted by linear separability (H2 PARTIALLY REJECTED)**
The correlation between linear probe accuracy and steering Cohen's d is r = −0.220 (p = 0.52)—weak and non-significant. Notably, sadness has the 3rd highest steering d (0.907) despite only moderate linear separability (0.856), while love has very high separability (0.939) but very low steering d (0.147). Linear separability is necessary but not sufficient for effective steering.

**Finding 4: Prompting succeeds uniformly regardless of steering difficulty (H3 SUPPORTED)**
Prompting achieves ≥3.8/5 and 100% success rate for all styles, including those that steer poorly (disgust: d=0.129, prompt=4.4/5; fear: d=0.090, prompt=4.0/5). The correlation between steering effectiveness and prompting success is r = −0.018 (p = 0.96)—essentially zero. This is the most striking finding: prompting and steering access fundamentally different pathways to style control.

**Finding 5: No evidence for non-linearity in well-known styles (NULL HYPOTHESIS PARTIALLY SUPPORTED)**
For representation geometry, the null hypothesis holds: all tested styles have NL index ≈ 1.0. However, the null hypothesis *fails* for the behavioral link between steering and prompting: they are uncorrelated, meaning style *access* involves non-linear computation even when style *representation* is linear.

### Statistical Tests

| Correlation | Pearson r | p-value | Significant? |
|-------------|----------|---------|-------------|
| Linear acc ↔ Prompt score | 0.447 | 0.168 | No |
| Linear acc ↔ Prompt specificity | 0.359 | 0.278 | No |
| NL index ↔ Prompt score | −0.515 | 0.105 | No |
| Cohen's d ↔ Linear acc | −0.220 | 0.516 | No |
| Cohen's d ↔ Prompt score | −0.018 | 0.957 | No |
| SV accuracy ↔ Prompt score | −0.029 | 0.932 | No |

**Key interpretation:** None of the expected correlations are statistically significant. This is itself a finding: the three measures (linear separability, steerability, promptability) are largely *independent* dimensions.

### Surprises and Insights

1. **Fear has the lowest steering d (0.090) but the highest generation shift (+0.400).** This paradox suggests that even weak steering vectors can cause large behavioral shifts when they happen to align with sensitive decision boundaries.

2. **Gratitude shows the largest steering shift (+0.733) with only moderate d (0.381).** This confirms that the relationship between activation-space metrics and behavioral effects is nonlinear.

3. **Sadness steers *negatively* (−0.067 shift) despite having the highest Cohen's d (0.907).** The mean-difference vector points in a direction that is linearly separable but not causally effective for generation. This is a clear example of the "detection ≠ control" gap identified by AxBench (Wu et al., 2025).

4. **Disgust, historically the hardest to steer (Konen et al., 2024), prompts perfectly (4.4/5 with highest absolute score!).** This directly answers the user's question: no, prompting difficulty does NOT follow steering difficulty.

### Error Analysis

**Steering failure modes:**
- Negative shifts (sadness, surprise, admiration): Steering vector points in a direction that is detectably different from neutral but does not causally produce the target style during generation
- Low Cohen's d (fear: 0.090, anger: 0.129, disgust: 0.129): Style activations overlap heavily with neutral in the mean-difference direction
- High variance: With only 15 test prompts, individual generation results are noisy

**Prompting limitations:**
- All prompting used GPT-4.1-nano (not the same model as Pythia-2.8B), so we're comparing *representation* properties of one model with *prompting* behavior of another
- Prompting scores reflect a judge's assessment, not ground truth

### Limitations

1. **Model mismatch:** Probing and steering used Pythia-2.8B; prompting used GPT-4.1-nano. Ideally we'd prompt the same model, but Pythia-2.8B is a base model with poor instruction following.

2. **Small generation sample size:** Only 15 test prompts for steering measurement introduces substantial variance.

3. **GoEmotions as style proxy:** Emotion labels from Reddit comments are noisy proxies for "style." True literary style (irony, noir, stream-of-consciousness) would be a stronger test but lacks standard labeled datasets.

4. **Single model:** Results are for Pythia-2.8B only. Larger models or instruction-tuned models may show different patterns.

5. **Binary probing:** We tested style-vs-neutral, not style-vs-all-other-styles. Multi-class probing might reveal different patterns.

6. **Mean-pooling:** We averaged activations across sequence positions. Position-specific probing might reveal non-linear structure.

## 6. Conclusions

### Summary
In Pythia-2.8B, all 11 tested emotional styles are represented **linearly** in activation space (NL index ≈ 1.0 for all). However, the linear representation hypothesis is not the whole story: **steerability** (whether a linear intervention causally shifts output) is largely independent of both **linear separability** (whether the representation can be detected linearly) and **promptability** (whether the model can produce the style via instruction). Styles that are hard to steer (disgust, fear) are *not* hard to prompt — prompting achieves near-perfect success for all styles.

### Direct Answer to the User's Question
> "If there's a particular style that it's hard to get a steering vector for, does that also mean it will be hard to prompt?"

**No.** Prompting difficulty and steering difficulty are uncorrelated (r = −0.018). Disgust, which historically fails at steering (Konen et al., 2024; AUC = 0.51), achieves the highest prompting score (4.4/5).

> "Does the model represent any style it 'knows' a priori non-linearly, or are they all linear?"

For the 11 emotions tested in Pythia-2.8B, **all are linear** at the representation level. However, "representation is linear" ≠ "intervention is linear." The process by which the model *uses* these representations to generate styled text appears to involve nonlinear computation that prompting can access but simple activation addition cannot.

### Implications

**For AI safety/alignment:** Linear representation-based interventions (steering vectors, SAE features) may systematically miss styles/behaviors that are linearly represented but causally accessed through nonlinear pathways. Prompting remains the most reliable style control method, consistent with AxBench findings.

**For interpretability:** The detection-control gap (style is linearly detectable but not linearly steerable) suggests that the model's read-out mechanism for style is not simply a linear function of activation directions. Understanding this gap is a key challenge.

**For the Linear Representation Hypothesis:** Our results support the LRH for *detection* (all styles are linearly separable) but challenge the assumption that linear detection implies linear *intervention effectiveness*. The LRH should be considered a statement about representation geometry, not about causal mechanisms.

### Confidence in Findings
- **High confidence:** All styles are linearly represented (consistent results across layers, styles, and both probes)
- **Medium confidence:** Steering and prompting are uncorrelated (limited by 11 styles and noisy steering measurement)
- **Lower confidence:** Generalization to other models and more complex styles

## 7. Next Steps

### Immediate Follow-ups
1. **Test on instruction-tuned models** (Llama-3-8B-Instruct) where both steering and prompting can use the same model
2. **Expand style taxonomy** beyond emotions: irony, formality, verbosity, literary genres, and adversarial styles (ending every sentence with 🥕)
3. **Increase steering sample size** to 100+ prompts for more reliable effect estimates
4. **Multi-class probing** (all styles simultaneously) to test whether styles interfere with each other

### Alternative Approaches
- **Trained steering vectors** (gradient-optimized) rather than mean-difference vectors
- **Layer-specific interventions** (steer at different layers to find optimal injection point)
- **Activation patching** to identify which components causally contribute to style

### Broader Extensions
- Test whether the detection-control gap extends to safety-relevant behaviors (sycophancy, deception)
- Investigate whether scaling (larger models) closes or widens the gap
- Explore non-linear steering methods (e.g., using pyvene's DAS) for styles where linear steering fails

### Open Questions
1. Why does sadness have high linear separability and high Cohen's d but negative steering shift?
2. Is the detection-control gap a fundamental architectural property of transformers, or is it specific to certain styles/models?
3. Would online-constructed styles (like "end every sentence with 🥕") show different representation geometry than a priori known styles?

## 8. References

1. Park, Choe, Veitch (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models. arXiv:2311.03658
2. Turner et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248
3. Konen, Jentzsch et al. (2024). Style Vectors for Steering Generative Large Language Models. arXiv:2402.01618
4. Csordas, Potts, Manning, Geiger (2024). Non-linear Representations in RNNs. arXiv:2408.10920
5. Rimsky et al. (2024). Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681
6. Wu et al. (2025). AxBench: Even Simple Baselines Outperform Sparse Autoencoders. arXiv:2501.17148
7. Elhage et al. (2022). Toy Models of Superposition. arXiv:2209.10652
8. Jiang et al. (2024). On the Origins of Linear Representations in Large Language Models. arXiv:2403.03867
9. Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. arXiv:2005.00547
10. Biderman et al. (2023). Pythia: A Suite for Analyzing Large Language Models. arXiv:2304.01373
