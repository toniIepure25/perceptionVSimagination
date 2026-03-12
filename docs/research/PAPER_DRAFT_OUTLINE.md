# Paper Draft Outline

> **Living document** — serves as a north-star for the narrative structure. Update as results emerge.

---

## Title

**The Semantic Skeleton of Imagination: Cross-Domain Neural Decoding Reveals Hierarchical Information Loss in Mental Imagery**

---

## Abstract (placeholder)

- **Context**: Brain-to-image decoding achieves impressive results for perception; mental imagery remains largely unexplored as a decoding target.
- **Gap**: Unknown whether perception-trained decoders transfer to imagery, and if not, *what specifically degrades* and why.
- **Approach**: Multi-target fMRI decoders (CLIP ViT-L/14 768-d, ViT-bigG/14 328K-d) trained on NSD perception, evaluated on NSD-Imagery, augmented with 19 novel analysis directions.
- **Key Finding**: (Expected) Imagery preserves semantic content while losing structural detail — a "semantic skeleton" pattern.
- **Significance**: Computational evidence for hierarchical predictive coding theories of imagination.

---

## 1. Introduction

### 1.1 Motivation

- Brain-to-image decoding as a window into neural representations (Ozcelik & VanRullen 2023; Scotti et al. 2023; Takagi & Nishimoto 2023)
- NSD dataset (Allen et al. 2022) as the standard benchmark
- Mental imagery as a related but distinct neural process (Pearson 2019; Kosslyn et al. 2006)
- NSD-Imagery extension as a naturalistic imagery benchmark

### 1.2 Research Questions

1. **Transfer gap**: Do perception-trained decoders work on imagery fMRI? How large is the drop?
2. **What degrades**: Uniform loss or hierarchical pattern?
3. **Model quality**: Do findings replicate across a weak (R@1=5.7%) and strong (R@1=58%) decoder?
4. **Bridging the gap**: Can lightweight adapters close it, and what does adapter structure reveal?

### 1.3 Contributions

1. First systematic evaluation of multi-model fMRI decoders on imagery data — including both lightweight (MLP, 6.3M params) and SoTA (vMF-NCE, 825M params) architectures
2. 19 novel analysis directions beyond accuracy metrics
3. Evidence for the "semantic skeleton" hypothesis
4. Reusable open-source framework with 51+ tests

---

## 2. Related Work

### 2.1 Brain-to-Image Decoding
- Ridge regression (Horikawa & Kamitani 2017), CLIP-guided (Ozcelik & VanRullen 2023), two-stage (Scotti et al. 2023), IP-Adapter (Ye et al. 2023)

### 2.2 Mental Imagery
- Shared substrates (Kosslyn et al. 2001), hierarchical differences (Dijkstra et al. 2019), predictive coding (Rao & Ballard 1999)

### 2.3 Representation Analysis Methods
- CKA (Kornblith et al. 2019), persistent homology (Carlsson 2009), VICReg (Bardes et al. 2022), Barlow Twins (Zbontar et al. 2021)

### 2.4 Domain Transfer
- Cross-subject (Defossez et al. 2023), LoRA (Hu et al. 2021), domain adversarial (Ganin et al. 2016)

---

## 3. Methods

### 3.1 Data
- **NSD Perception**: 30K trials, 3 repetitions per stimulus, subj01 (Allen et al. 2022)
- **NSD-Imagery**: Subset of NSD stimuli recalled from memory under controlled conditions
- **Preprocessing**: Z-score per session + PCA(3072) for this project; raw 15,724 voxels for FMRI2images

### 3.2 Decoder Architectures

#### 3.2.1 Lightweight Decoder (This Project)
- **Input**: 3,072-d PCA features
- **CLIP target**: ViT-L/14 768-d CLS token
- **Architectures**: Ridge, MLP (~6.3M), TwoStage (~7.9M)
- **Best perception**: R@1=5.7%, cosine=0.81

#### 3.2.2 High-Capacity Decoder (FMRI2images)
- **Input**: 15,724 raw voxels (nsdgeneral)
- **CLIP target**: ViT-bigG/14, 1280-d × 257 tokens = 328,960-d
- **Architecture**: 4-layer residual MLP → vMF decoder (825M params)
- **Training**: vMF-NCE + SoftCLIP + MixCo + EMA
- **Best perception**: R@1~58%, CSLS R@1~70%

### 3.3 Cross-Domain Evaluation Protocol
1. Train all decoders exclusively on perception fMRI
2. Evaluate on held-out perception trials (within-domain baseline)
3. Evaluate on imagery fMRI (cross-domain transfer)
4. Compare with both weak and strong decoders
5. Metrics: CLIP cosine, R@K, noise-ceiling normalized

### 3.4 Imagery Adapters
- LinearAdapter, MLPAdapter (frozen backbone)
- LoRA (ranks 2-16) for parameter-efficient transfer
- Domain adversarial (DANN) for domain-invariant features

### 3.5 Novel Analysis Methods (19 Directions)

#### Tier 1: Core (Directions 1-6)
1. **Dimensionality Gap** — PCA participation ratio, intrinsic dimensionality
2. **Uncertainty as Vividness** — MC Dropout variance ↔ decoding accuracy
3. **Semantic Survival** — Per-concept preservation ratios (category, scene, texture)
4. **Topological RSA** — Persistent homology H0/H1 between perception/imagery RDMs
5. **Individual Fingerprints** — Second-order RSA of degradation profiles
6. **Semantic-Structural Dissociation** — SSI: CLIP preservation vs. structural preservation

#### Tier 2: Reality Perception (Directions 7-10)
7. **Reality Monitor** — PRM theory: metacognitive confusability prediction
8. **Reality Confusion** — Decision boundary estimation in embedding space
9. **Adversarial Reality** — GAN-style discriminator accuracy at perception/imagery classification
10. **Hierarchical Reality** — Layer-by-layer emergence of the perception-imagery gap

#### Tier 3: Generative Imagination (Directions 11-15)
11. **Compositional Imagination** — Brain algebra (combine concepts in embedding space)
12. **Predictive Coding** — Top-down vs bottom-up information flow indices
13. **Manifold Geometry** — Centrality bias (imagined representations drift toward manifold center)
14. **Modality Decomposition** — Shared core vs modality-specific residual components
15. **Creative Divergence** — Systematic transformation rules of imagination

#### Cross-Cutting
- **CKA** — Layer-wise representational similarity across conditions
- **UMAP/t-SNE** — Manifold visualization with density comparison
- **ROI Decoding** — Per-brain-region (V1-V4, ventral temporal, parietal) accuracy
- **Interpretability** — Integrated Gradients, SmoothGrad, Grad×Input attribution
- **Noise-Ceiling Normalization** — All metrics as % of theoretical maximum
- **SoTA Comparison** — Against 8 published baselines with LaTeX tables

---

## 4. Expected Results

### 4.1 Transfer Gap (H1)
Perception-trained decoders show 15-30% CLIP cosine reduction on imagery. Larger drops for structural metrics. The weak decoder (5.7% R@1) shows proportionally larger gaps than the strong decoder (58% R@1).

### 4.2 Dimensionality Compression
Imagery embeddings exhibit 20-40% lower effective dimensionality — imagery is a lossy compression that discards fine-grained variation.

### 4.3 Uncertainty-Vividness Link
MC Dropout uncertainty negatively correlates with decoding accuracy (ρ ≈ -0.3 to -0.5), stronger for imagery than perception.

### 4.4 Semantic Survival Gradient
Abstract categories >80% preservation, scene features ~60-70%, low-level texture <40% — the "semantic skeleton."

### 4.5 Topological Reorganization
H0 (clusters) preserved; H1 (loops/fine relationships) disrupted in imagery.

### 4.6 Subject-Specific Signatures
Degradation profiles more similar within-subject than between-subjects.

### 4.7 Cross-Model Consistency
If both the weak (6.3M) and strong (825M) decoders show the same hierarchical pattern, the finding reflects neural organization, not model limitations.

---

## 5. Discussion

### 5.1 The Semantic Skeleton Hypothesis
Mental imagery constructs a sparse, high-level representation preserving categorical structure while losing spatial/textural detail.

### 5.2 Predictive Coding Implications
Consistent with hierarchical predictive coding: higher layers generate internal representations without bottom-up input.

### 5.3 Cross-Model Robustness
Same patterns across weak and strong decoders → genuine neural phenomenon. Different patterns → model quality mediates the finding (itself valuable).

### 5.4 Limitations
- NSD-Imagery sample sizes may limit statistical power
- Ground truth for imagery is the original perceived image, not the actual mental image
- Results may be specific to NSD natural scenes

---

## 6. Conclusion

[To be written after results.]

---

## References

- Allen et al. (2022). *Nature Neuroscience*.
- Bardes et al. (2022). VICReg. *ICML*.
- Carlsson (2009). Topology and data. *Bull. AMS*.
- Defossez et al. (2023). *Nature Machine Intelligence*.
- Dijkstra et al. (2019). *Trends in Cognitive Sciences*.
- Gal & Ghahramani (2016). Dropout as Bayesian. *ICML*.
- Ganin et al. (2016). Domain adversarial. *JMLR*.
- Hu et al. (2021). LoRA. *ICLR*.
- Kornblith et al. (2019). CKA. *ICML*.
- Kosslyn et al. (2001). Neural foundations. *Nature Reviews Neuroscience*.
- Ozcelik & VanRullen (2023). Brain-Diffuser. *Scientific Reports*.
- Pearson (2019). *Nature Reviews Neuroscience*.
- Rao & Ballard (1999). *Nature Neuroscience*.
- Scotti et al. (2023). MindEye. *NeurIPS*.
- Takagi & Nishimoto (2023). *CVPR*.
- Ye et al. (2023). IP-Adapter. *arXiv*.
- Zbontar et al. (2021). Barlow Twins. *ICML*.
