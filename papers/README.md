# Downloaded Papers

## Core Papers (22 total)

### Linear Representation Hypothesis & Geometry

1. **The Linear Representation Hypothesis and the Geometry of Large Language Models** (Park et al., 2023)
   - File: `2311.03658_The_Linear_Representation_Hypothesis_and_the_Geometry_of_Lar.pdf`
   - arXiv: 2311.03658 | Citations: 408 | ICML 2024
   - Why relevant: Foundational formalization of LRH. Tests 27 concepts; finds 1 exception (thing→part). Core theoretical framework for our research.

2. **On the Origins of Linear Representations in Large Language Models** (Jiang et al., 2024)
   - File: `2403.03867_On_the_Origins_of_Linear_Representations_in_Large_Language_M.pdf`
   - arXiv: 2403.03867 | Citations: 61
   - Why relevant: Theoretical explanation for why linearity emerges. Assumes binary concepts — non-binary/compositional concepts (like style) may not satisfy conditions.

3. **The Geometry of Categorical and Hierarchical Concepts in LLMs** (Park et al., 2024)
   - File: `2406.01506_The_Geometry_of_Categorical_and_Hierarchical_Concepts_in_Lar.pdf`
   - arXiv: 2406.01506 | Citations: 86 | ICLR 2025
   - Why relevant: Extends LRH to polytope/subspace structures. Style is inherently categorical and hierarchical.

4. **Emergent Linear Representations in World Models** (Nanda et al., 2023)
   - File: `2309.00941_Emergent_Linear_Representations_in_World_Models_of_Self_Supe.pdf`
   - arXiv: 2309.00941 | Citations: 290
   - Why relevant: Evidence for linearity in sequence models, but limited to well-defined game states.

5. **Language Models Represent Space and Time** (Gurnee & Tegmark, 2023)
   - File: `2310.02207_Language_Models_Represent_Space_and_Time.pdf`
   - arXiv: 2310.02207 | Citations: 280
   - Why relevant: Linear spatial/temporal representations. Concrete concepts — style is more abstract.

### Steering Vectors & Activation Engineering

6. **Steering Language Models With Activation Engineering** (Turner et al., 2023)
   - File: `2308.10248_Steering_Language_Models_With_Activation_Engineering.pdf`
   - arXiv: 2308.10248 | Citations: 400
   - Why relevant: Introduces ActAdd. "Art" topic shows zero steering effect. Non-monotonic partial dimension results.

7. **Steering Llama 2 via Contrastive Activation Addition** (Rimsky et al., 2024)
   - File: `2312.06681_Steering_Llama_2_via_Contrastive_Activation_Addition.pdf`
   - arXiv: 2312.06681 | Citations: 579 | ACL 2024
   - Why relevant: Scales CAA to 7 behavioral dimensions. Works for binary behavioral concepts.

8. **Extracting Latent Steering Vectors from Pretrained LMs** (Subramani et al., 2022)
   - File: `2205.05124_Extracting_Latent_Steering_Vectors_from_Pretrained_Language.pdf`
   - arXiv: 2205.05124 | Citations: 167
   - Why relevant: Pioneer work on optimization-based steering vectors. Each sentence needs its own vector.

9. **Improving Activation Steering with Mean-Centring** (2023)
   - File: `2312.03813_Improving_Activation_Steering_in_Language_Models_with_Mean_C.pdf`
   - arXiv: 2312.03813 | Citations: 59
   - Why relevant: Improved steering technique — still assumes linear representations.

10. **Word Embeddings Are Steers for Language Models** (Han et al., 2023)
    - File: `2305.12798_Word_Embeddings_Are_Steers_for_Language_Models.pdf`
    - arXiv: 2305.12798 | Citations: 77
    - Why relevant: Uses d×d matrix for style control — implies single direction insufficient.

### Style-Specific Papers

11. **Style Vectors for Steering Generative Large Language Models** (Konen et al., 2024)
    - File: `2402.01618_Style_Vectors_for_Steering_Generative_Large_Language_Models.pdf`
    - arXiv: 2402.01618 | Citations: 55 | EACL 2024 Findings
    - Why relevant: **Most directly relevant.** Tests emotion and writing style steering. Disgust/surprise fail.

12. **Personalized Steering of LLMs: Versatile Steering Vectors via BPO** (2024)
    - File: `2406.00045_Personalized_Steering_of_Large_Language_Models__Versatile_St.pdf`
    - arXiv: 2406.00045 | Citations: 87
    - Why relevant: Personalized style steering via preference optimization.

### Non-Linear Representations & Superposition

13. **Recurrent Neural Networks Learn Non-Linear Representations** (Csordas et al., 2024)
    - File: `2408.10920_Recurrent_Neural_Networks_Learn_to_Store_and_Generate_Sequen.pdf`
    - arXiv: 2408.10920 | Citations: 39
    - Why relevant: **Key counterexample to strong LRH.** "Onion representations" — magnitude-based encoding. Linear methods score 0-1% where onion interventions score 83-87%.

14. **Toy Models of Superposition** (Elhage et al., 2022)
    - File: `2209.10652_Toy_Models_of_Superposition.pdf`
    - arXiv: 2209.10652 | Citations: 664
    - Why relevant: Foundational work on superposition. Sparse/low-importance features are polysemantic.

15. **Polysemanticity and Capacity in Neural Networks** (Scherlis et al., 2022)
    - File: `2210.01892_Polysemanticity_and_Capacity_in_Neural_Networks.pdf`
    - arXiv: 2210.01892 | Citations: 54
    - Why relevant: Feature capacity framework. Less important features → polysemantic → harder to isolate.

### Benchmarks & Evaluation

16. **AxBench: Steering LLMs?** (Wu et al., 2025)
    - File: `2501.17148_AxBench__Steering_LLMs__Even_Simple_Baselines_Outperform_Spa.pdf`
    - arXiv: 2501.17148 | Citations: 130
    - Why relevant: Prompting outperforms all steering methods. Key benchmark for comparing approaches.

17. **A Unified Understanding and Evaluation of Steering Methods** (2025)
    - File: `2502.14829_A_Unified_Understanding_and_Evaluation_of_Steering_Methods.pdf`
    - arXiv: 2502.14829 | Citations: 27
    - Note: PDF may contain wrong content (file mismatch detected)

### Advanced Steering Methods

18. **Plug and Play Language Models** (Dathathri et al., 2019)
    - File: `1912.02164_Plug_and_Play_Language_Models__A_Simple_Approach_to_Controll.pdf`
    - arXiv: 1912.02164 | Citations: 1138
    - Why relevant: Early gradient-based controlled generation. Baseline approach.

19. **Inference-Time Intervention** (Li et al., 2023)
    - File: `2306.03341_Inference_Time_Intervention__Eliciting_Truthful_Answers_from.pdf`
    - arXiv: 2306.03341 | Citations: 951
    - Why relevant: Probing + intervention for truthfulness. Linear probing identifies "truthful" directions.

20. **Steering Without Side Effects** (Stickland et al., 2024)
    - File: `2406.15518_Steering_Without_Side_Effects__Improving_Post_Deployment_Con.pdf`
    - arXiv: 2406.15518 | Citations: 43
    - Why relevant: Documents that steering causes capability degradation — fundamental limit of linear approach.

21. **Improving Steering Vectors by Targeting SAE Features** (Chalnev et al., 2024)
    - File: `2411.02193_Improving_Steering_Vectors_by_Targeting_Sparse_Autoencoder_F.pdf`
    - arXiv: 2411.02193 | Citations: 55
    - Why relevant: Even SAE-based steering produces unexpected side effects. Features are entangled.

22. **The Geometry of Concepts: SAE Feature Structure** (Li et al., 2024)
    - File: `2410.19750_The_Geometry_of_Concepts__Sparse_Autoencoder_Feature_Structu.pdf`
    - arXiv: 2410.19750 | Citations: 41
    - Why relevant: SAE features form complex multi-scale geometric structures (crystals, lobes) — not purely linear.
