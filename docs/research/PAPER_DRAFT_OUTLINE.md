# Paper Draft Outline

> **Living document** -- serves as a north-star for the narrative structure of the research paper. Update as results emerge.

---

## Title

**The Semantic Skeleton of Imagination: Cross-Domain Neural Decoding Reveals Hierarchical Information Loss in Mental Imagery**

---

## Abstract (placeholder structure)

- **Context**: Brain-to-image decoding has achieved impressive results for perception, but mental imagery remains unexplored as a decoding target.
- **Gap**: It is unknown whether perception-trained decoders transfer to imagery, and if not, *what specifically degrades* and why.
- **Approach**: We use multi-target fMRI decoders (CLIP, IP-Adapter, SD VAE) trained on the Natural Scenes Dataset (NSD) and evaluate them on the NSD-Imagery benchmark, augmented with six novel analysis directions.
- **Key Finding**: Imagery preserves high-level semantic content (CLIP embeddings) while losing structural detail (SD-latent, IP-Adapter tokens), producing a characteristic "semantic skeleton" -- a compressed, lower-dimensional representation that maintains categorical boundaries but drops spatial layout.
- **Significance**: This dissociation provides computational evidence for hierarchical predictive coding theories of imagination and suggests that mental imagery engages a fundamentally different generative process than bottom-up perception.

---

## 1. Introduction

### 1.1 Motivation

- Brain-to-image decoding as a window into neural representations (Ozcelik & VanRullen 2023; Scotti et al. 2023; Takagi & Nishimoto 2023)
- The NSD dataset (Allen et al. 2022) as the standard benchmark for visual decoding
- Mental imagery as a related but distinct neural process (Pearson 2019; Kosslyn et al. 2006)
- The NSD-Imagery extension (hypothetical benchmark) as a naturalistic imagery dataset

### 1.2 Research Questions

1. **Transfer gap**: Do perception-trained decoders work on imagery fMRI? How large is the performance drop?
2. **What degrades**: Is the loss uniform across representational levels, or does it follow a hierarchical pattern?
3. **Bridging the gap**: Can lightweight adapters close the transfer gap, and what does their structure reveal?

### 1.3 Contributions

1. First systematic evaluation of multi-target fMRI decoders on imagery data
2. Six novel analysis directions that go beyond accuracy metrics to characterize *how* representations change
3. Evidence for the "semantic skeleton" hypothesis: imagery is a low-dimensional, semantics-preserving projection of perceptual space
4. A reusable open-source framework for cross-domain fMRI decoding

---

## 2. Related Work

### 2.1 Brain-to-Image Decoding

- Ridge regression approaches (Horikawa & Kamitani 2017)
- CLIP-guided decoding (Ozcelik & VanRullen 2023)
- Two-stage architectures: embedding prediction + conditional generation (Scotti et al. 2023)
- IP-Adapter conditioning (Ye et al. 2023)

### 2.2 Mental Imagery and Neural Representations

- Shared neural substrates for perception and imagery (Kosslyn et al. 2001)
- Hierarchical differences: imagery engages top-down pathways (Dijkstra et al. 2019)
- Predictive coding accounts (Rao & Ballard 1999; Keller & Mrsic-Flogel 2018)

### 2.3 Domain Transfer in Neural Decoding

- Cross-subject transfer (Defossez et al. 2023)
- Limited work on perception-to-imagery transfer

---

## 3. Methods

### 3.1 Data

- **NSD Perception**: 10,000 images, 3 repetitions per stimulus, 8 subjects (Allen et al. 2022)
- **NSD-Imagery**: Subset of NSD stimuli recalled from memory under controlled conditions
- Preprocessing: denoising, ROI selection (V1-V4, higher visual cortex, ventral temporal)

### 3.2 Multi-Target Decoder Architecture

- **CLIP pathway**: Linear projection from fMRI voxels to CLIP ViT-L/14 embedding space (768-d)
- **IP-Adapter pathway**: MLP mapping to 4 IP-Adapter tokens (4 × 768-d), capturing fine-grained visual detail
- **SD VAE pathway**: Linear projection to Stable Diffusion VAE latent space (4 × 64 × 64), encoding coarse spatial structure
- **MultiTargetDecoder**: Shared trunk with task-specific heads, trained with composite loss

### 3.3 Cross-Domain Evaluation Protocol

- Train all decoders exclusively on perception fMRI
- Evaluate on held-out perception trials (within-domain baseline)
- Evaluate on imagery fMRI (cross-domain transfer)
- Metrics: CLIP cosine similarity, retrieval@K, pixel-level SSIM for reconstructions

### 3.4 Imagery Adapter

- Lightweight modules (LinearAdapter, MLPAdapter) that remap perception-space outputs to imagery-space
- Frozen perception backbone + trainable adapter (parameter-efficient)
- Ablation: linear vs. MLP, frozen vs. fine-tuned backbone

### 3.5 Novel Analysis Methods

#### 3.5.1 Direction 1: Representational Dimensionality

- PCA participation ratio (Gao et al. 2017) on perception vs. imagery embeddings
- Intrinsic dimensionality estimation (Levina & Bickel 2004)
- Hypothesis: imagery embeddings occupy a lower-dimensional subspace

#### 3.5.2 Direction 2: Uncertainty as Vividness Proxy

- MC Dropout at inference time (Gal & Ghahramani 2016) to obtain predictive uncertainty
- Correlate trial-level uncertainty with decoding accuracy
- Hypothesis: high-confidence imagery trials decode better (proxy for subjective vividness)

#### 3.5.3 Direction 3: Semantic Survival Analysis

- Project CLIP embeddings onto concept axes (object category, scene type, emotional valence)
- Compare per-concept preservation ratios between perception and imagery
- Hypothesis: abstract/semantic concepts survive imagery; texture/color concepts degrade

#### 3.5.4 Direction 4: Topological RSA

- Compute persistence diagrams (H0, H1) from representational distance matrices (Carlsson 2009)
- Wasserstein distance between perception and imagery persistence diagrams
- Separate analysis for metric geometry (RDM correlation) vs. topological structure (persistent homology)
- Hypothesis: metric geometry changes while topological structure partially survives

#### 3.5.5 Direction 5: Individual Difference Fingerprints

- Compute per-subject degradation profiles across all metrics
- Second-order RSA: do subjects who are similar in perception remain similar in imagery?
- Adapter weight analysis: do architecturally similar adapters emerge for similar subjects?
- Hypothesis: the perception-imagery gap has a stable, subject-specific signature

#### 3.5.6 Direction 6: Semantic-Structural Dissociation

- Compare transfer performance across the three decoder targets:
  - CLIP (high-level semantics)
  - IP-Adapter tokens (mid-level visual features)
  - SD VAE latents (low-level spatial structure)
- Compute Semantic-Structural Index (SSI): ratio of CLIP preservation to SD-latent preservation
- Hypothesis: SSI >> 1, confirming that semantics survive while structure degrades

---

## 4. Expected Results

### 4.1 Transfer Gap (Baseline)

We expect perception-trained decoders to show a significant but non-catastrophic drop on imagery data: 15-30% reduction in CLIP cosine similarity, with larger drops for structural metrics (SSIM, SD-latent MSE). This establishes the baseline transfer gap.

### 4.2 Dimensionality Compression

Imagery representations should exhibit 20-40% lower effective dimensionality (participation ratio) than perception, consistent with the idea that imagery is a lossy compression of the perceptual manifold that discards fine-grained variation.

### 4.3 Uncertainty-Vividness Link

MC Dropout uncertainty should negatively correlate with single-trial decoding accuracy (ρ ≈ -0.3 to -0.5), and this correlation should be stronger for imagery than perception, suggesting that uncertainty captures variance in imagery quality.

### 4.4 Semantic Survival Gradient

We expect a clear hierarchy: abstract categories (animal vs. vehicle) will show >80% preservation, scene-level features (indoor vs. outdoor) ~60-70%, and low-level features (texture, color) <40%. This produces the "semantic skeleton" pattern.

### 4.5 Topological Reorganization

Persistent homology should reveal that 0th-order topology (connected components = category clusters) is largely preserved in imagery, while 1st-order topology (loops = fine-grained relationships) is disrupted. This provides a formal, metric-free characterization of representational change.

### 4.6 Subject-Specific Signatures

Cross-subject analysis should reveal that degradation profiles are more similar within-subject (across sessions) than between-subjects, and adapter weight similarity should correlate with behavioral similarity in imagery tasks. This suggests the perception-imagery mapping is idiosyncratic.

### 4.7 Semantic-Structural Dissociation

The SSI should be significantly greater than 1 (expected range: 2-5), confirming that CLIP-level semantics are preferentially preserved over SD-latent-level structure. This is the central result tying all directions together.

---

## 5. Discussion

### 5.1 The Semantic Skeleton Hypothesis

Synthesis of all six directions into a unified narrative: mental imagery constructs a "semantic skeleton" -- a sparse, high-level representation that preserves categorical and conceptual structure while losing the fine-grained spatial, textural, and colorimetric detail that characterizes veridical perception. This is not a uniform degradation but a selective, hierarchical compression.

### 5.2 Implications for Predictive Coding

The dissociation between semantic and structural preservation is consistent with hierarchical predictive coding: imagination engages top-down generative predictions that capture the statistical structure of the world (what) but not its precise realization (how). The higher cortical layers that encode abstract semantics can generate internal representations without bottom-up sensory input, while lower layers that encode pixel-level structure require veridical input.

### 5.3 Computational Implications

The adapter framework reveals that the perception-to-imagery mapping is surprisingly low-rank: a small linear or shallow MLP transform suffices to bridge the gap. This suggests that imagery representations live in a subspace of perceptual space, not in a qualitatively different space.

### 5.4 Limitations

- NSD-Imagery sample sizes may limit statistical power for individual-difference analyses
- The "ground truth" for imagery is the originally perceived image, which may not match the actual mental image
- Topological analyses require careful hyperparameter tuning (persistence threshold, distance metric)
- Results may be specific to the NSD stimulus set (natural scenes) and not generalize to other imagery types

### 5.5 Future Directions

- Temporal dynamics: use MEG or EEG to track the time course of semantic vs. structural information
- Intervention: use TMS to test causal role of specific cortical regions in the dissociation
- Clinical applications: imagery quality in aphantasia, PTSD, and synesthesia
- Generative models: train imagery-specific diffusion models that accept the "semantic skeleton" as conditioning

---

## 6. Conclusion

[To be written after results are obtained.]

---

## References (key papers to cite)

- Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*.
- Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*.
- Defossez, A., et al. (2023). Decoding speech from non-invasive brain recordings. *Nature Machine Intelligence*.
- Dijkstra, N., Bosch, S. E., & van Gerven, M. A. (2019). Shared neural mechanisms of visual perception and imagery. *Trends in Cognitive Sciences*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Gao, P., et al. (2017). A theory of multineuronal dimensionality, dynamics and measurement. *bioRxiv*.
- Keller, G. B., & Mrsic-Flogel, T. D. (2018). Predictive processing: a canonical cortical computation. *Neuron*.
- Kosslyn, S. M., Ganis, G., & Thompson, W. L. (2001). Neural foundations of imagery. *Nature Reviews Neuroscience*.
- Kosslyn, S. M., Thompson, W. L., & Ganis, G. (2006). *The Case for Mental Imagery*. Oxford University Press.
- Levina, E., & Bickel, P. (2004). Maximum likelihood estimation of intrinsic dimension. *NeurIPS*.
- Ozcelik, F., & VanRullen, R. (2023). Natural scene reconstruction from fMRI signals using generative latent diffusion. *Scientific Reports*.
- Pearson, J. (2019). The human imagination: the cognitive neuroscience of visual mental imagery. *Nature Reviews Neuroscience*.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Scotti, P., et al. (2023). Reconstructing the mind's eye: fMRI-to-image with contrastive learning and diffusion priors. *NeurIPS*.
- Takagi, Y., & Nishimoto, S. (2023). High-resolution image reconstruction with latent diffusion models from human brain activity. *CVPR*.
- Ye, H., et al. (2023). IP-Adapter: text compatible image prompt adapter for text-to-image diffusion models. *arXiv*.
