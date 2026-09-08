# Chapter 22 research notes

Route: chapter authored by the assigned high-reasoning author after bounded source verification by `gpt-5.6-luna` at high reasoning. Verification date: 2026-09-08. Code, vectors, and arithmetic are course-authored.

- Hinton and Salakhutdinov, “Reducing the Dimensionality of Data with Neural Networks,” Science 313 (2006), https://doi.org/10.1126/science.1127647. Supports encoder–decoder dimensionality reduction and gradient fine-tuning.
- Vincent et al., “Stacked Denoising Autoencoders,” JMLR 11 (2010), https://www.jmlr.org/papers/v11/vincent10a.html. Supports reconstructing clean inputs from corrupted views and evaluating learned features downstream.
- Chen et al., “A Simple Framework for Contrastive Learning of Visual Representations,” ICML 2020, https://proceedings.mlr.press/v119/chen20j.html. Supports the importance of view construction, a contrastive objective, and evaluating representations separately from pretraining loss.
- van den Oord, Li, and Vinyals, “Representation Learning with Contrastive Predictive Coding,” arXiv:1807.03748, https://arxiv.org/abs/1807.03748. Supports noise-contrastive prediction of future latent representations. This is an arXiv primary preprint.

The executable uses analytic gradients through both sides of a shared normalized encoder and stable InfoNCE logits. Its six-pair retrieval result is training-set evidence only.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 20–24; the researcher supplied checks and source findings, while Astra implemented the corrections.

Confirmed analytic gradients through both shared contrastive branches and the autoencoder. Corrected stale numerical-training wording; retained numerical gradient checks. Reordered loss arithmetic and added non-finite input/tiny-temperature rejection tests. Training-pair retrieval is not a transfer estimate.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
