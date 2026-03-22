# Research Plan: Non-linear Diction

## Motivation & Novelty Assessment

### Why This Research Matters
Steering vectors have become a standard tool for controlling LLM behavior, but they rest on the Linear Representation Hypothesis—that concepts (including styles) correspond to directions in activation space. If some styles are encoded non-linearly, linear steering will systematically fail for them. Understanding which styles resist linear control, and whether this correlates with prompting difficulty, has direct implications for controllable generation, AI safety (alignment via representation engineering), and our theoretical understanding of how LLMs encode linguistic knowledge.

### Gap in Existing Work
Based on the literature review, three critical gaps exist:
1. **No systematic study of which styles resist linear steering.** Papers test a handful of styles; none evaluate linearity across a broad style taxonomy.
2. **No comparison of steering difficulty vs. prompting difficulty.** Our hypothesis specifically links these—if a style can't be steered, can it be prompted? No paper tests this.
3. **Non-linear representations studied only in small synthetic settings.** The onion representation paper (Csordas et al., 2024) uses small GRUs. Whether similar non-linear encodings exist for style in large transformers is unexplored.

### Our Novel Contribution
We conduct the first systematic comparison of:
- **Linear separability** (linear probe accuracy) vs. **non-linear separability** (MLP probe accuracy) across 10+ styles
- **Steering vector effectiveness** for each style
- **Prompting effectiveness** for each style (via real LLM API calls)
- **The correlation between all three**, testing whether styles that are hard to steer are also hard to prompt, and whether this is predicted by non-linear encoding.

### Experiment Justification
- **Experiment 1 (Probing):** Measures whether style information is linearly accessible in the model's activations. The ratio of MLP-to-linear probe accuracy gives a "non-linearity index" for each style. This is the foundational measurement.
- **Experiment 2 (Steering):** Tests whether linearly-encoded styles can be steered with contrastive activation addition (CAA). If linear probes succeed but steering fails, it suggests the representation is read-only linear but not causally linear.
- **Experiment 3 (Prompting):** Tests whether the model can produce each style via prompting. If prompting succeeds where steering fails, the model accesses the style through non-linear computation paths.
- **Experiment 4 (Correlation):** The core hypothesis test—correlating steering difficulty, prompting difficulty, and non-linearity index across styles.

## Research Question
Does the model represent any style it "knows" a priori non-linearly? Specifically: if a style is difficult to obtain a steering vector for, is it also difficult to prompt for? Or does prompting access non-linear pathways that steering vectors cannot?

## Hypothesis Decomposition
- **H1:** Some styles are encoded more non-linearly than others (MLP probe >> linear probe).
- **H2:** Styles with higher non-linearity index are harder to steer with linear vectors.
- **H3:** Styles that are hard to steer are NOT necessarily hard to prompt (prompting can access non-linear representations).
- **H4:** Simple binary styles (positive/negative sentiment) are linearly encoded; complex/compositional styles (disgust, irony, formality gradients) are more non-linear.
- **Null hypothesis:** All styles the model "knows" are linearly represented, and steering difficulty correlates perfectly with prompting difficulty.

## Proposed Methodology

### Approach
Use Gemma-2-2B (well-supported by TransformerLens, fits easily on our GPUs) to:
1. Collect activations for texts labeled with different styles from GoEmotions (emotions) and custom style prompts (writing styles)
2. Train linear and MLP probes on these activations at multiple layers
3. Extract CAA steering vectors for each style
4. Measure steering success via a classifier
5. Measure prompting success via real LLM API calls (GPT-4.1 via OpenRouter)
6. Correlate all measurements

### Model Choice
- **Primary:** Gemma-2-2B — fast inference, TransformerLens support, fits on single GPU
- **Classifier for evaluation:** A separate RoBERTa-based emotion classifier (or GPT-4.1 as judge)

### Styles to Test (10+ categories)
From GoEmotions (known difficulty spectrum):
1. Joy (easy — known to work)
2. Anger (easy)
3. Sadness (medium)
4. Fear (medium)
5. Disgust (hard — known to fail in Konen et al.)
6. Surprise (hard — known to fail)
7. Love (medium)
8. Gratitude (medium)

Custom writing styles:
9. Formal/academic
10. Sarcastic/ironic
11. Poetic/literary
12. Terse/telegraphic

### Experimental Steps
1. Load GoEmotions dataset, filter to styles with ≥200 examples each
2. Load Gemma-2-2B with TransformerLens
3. Run texts through model, cache activations at layers [4, 8, 12, 16, 20, 24]
4. For each style × layer: train linear probe and 2-layer MLP probe (5-fold CV)
5. Compute non-linearity index = MLP_acc / linear_acc for each style
6. Extract CAA vectors: mean(style_activations) - mean(neutral_activations) at best probing layer
7. Generate text with and without steering vectors; classify output
8. Generate text with style prompts via API; classify output
9. Correlate steering success, prompting success, and non-linearity index

### Baselines
- **Random vector steering** (control for steering specificity)
- **Linear probe accuracy** (ceiling for linear methods)
- **Unsteered generation** (baseline style distribution)

### Evaluation Metrics
- **Probe accuracy** (linear and MLP, 5-fold CV)
- **Non-linearity index** = MLP_acc / linear_acc
- **Steering success rate** = fraction of steered outputs classified as target style
- **Prompting success rate** = fraction of prompted outputs classified as target style
- **Pearson/Spearman correlation** between steering difficulty and prompting difficulty
- **Pearson/Spearman correlation** between non-linearity index and steering difficulty

### Statistical Analysis Plan
- 5-fold cross-validation for all probe accuracies
- Bootstrap confidence intervals (1000 resamples) for steering/prompting success rates
- Pearson and Spearman correlations with p-values for the core hypothesis tests
- Significance level: α = 0.05
- Multiple comparison correction: Bonferroni for pairwise style comparisons

## Expected Outcomes
- **If H1-H3 supported:** Some styles (disgust, surprise, irony) show high non-linearity index, low steering success, but moderate-to-high prompting success → the model encodes these non-linearly and accesses them through non-linear computation.
- **If null hypothesis supported:** All styles show similar non-linearity indices, and steering/prompting difficulty are perfectly correlated → linear representations suffice for all styles.

## Timeline and Milestones
1. Planning: 20 min ✓
2. Environment setup + data preparation: 15 min
3. Activation collection + probing: 30 min
4. Steering experiments: 30 min
5. Prompting experiments: 20 min
6. Analysis + visualization: 30 min
7. Documentation: 20 min

## Potential Challenges
- **GoEmotions label noise:** Multi-label, crowd-sourced → some noise. Mitigate by requiring high-confidence labels.
- **Gemma-2-2B may not "know" all styles equally well.** Mitigate by including styles we expect to work (sentiment) as positive controls.
- **Classifier evaluation may not perfectly capture style.** Use multiple evaluation approaches (classifier + LLM-as-judge).
- **Small sample sizes for rare emotions.** Filter to emotions with ≥200 examples.

## Success Criteria
1. Successfully measure non-linearity index for ≥8 styles
2. Successfully measure steering success for ≥8 styles
3. Successfully measure prompting success for ≥8 styles
4. Produce a correlation analysis with statistical significance tests
5. Clear visualization of the relationship between linearity, steerability, and promptability
