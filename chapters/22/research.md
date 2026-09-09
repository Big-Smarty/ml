# Chapter 22: redesign research and implementation record

Author route: one GPT-6 Astra High section owner, 2026-09-09. Read the complete previous lesson, metadata, research notes, reference and starter sources before redesign. The active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract replaces historical Rustlings and broken-starter requirements. Reference projects remain unchanged; tested numerical algorithms are adapted into labs/s04-deep-learning with learner-selected cores and separate explained solutions.

## Evidence and source claims

The following primary sources were previously verified in the course's 2026-09-08 research and independent review. This redesign reuses those bounded claims; it does not claim a new literature search or reproduction of the papers' full experiments.

- Reducing the Dimensionality of Data with Neural Networks — https://doi.org/10.1126/science.1127647
  Primary deep-autoencoder dimensionality-reduction paper.
- Stacked Denoising Autoencoders — https://www.jmlr.org/papers/v11/vincent10a.html
  Primary account of reconstructing clean inputs from corrupted views.
- A Simple Framework for Contrastive Learning of Visual Representations — https://proceedings.mlr.press/v119/chen20j.html
  Primary SimCLR paper on view construction, projection heads, and contrastive training.
- Representation Learning with Contrastive Predictive Coding — https://arxiv.org/abs/1807.03748
  Primary CPC preprint introducing a contrastive predictive objective.

## Active learning route

- Session 1: Implement AE encoder derivatives through the old decoder and tanh.
- Session 2: Accumulate both shared-encoder paths and verify an asymmetric paired batch.
- Session 3: Implement epsilon-aware cosine neighbors and compare objectives with retrieval.

Every session has worked values, named Rust work, a checkpoint or controlled experiment, optional explanation and a changed-input transfer task. Baseline success demonstrates a functioning earlier model, not the completed learning goal. The chapter's --check calls the selected learner core. Intentional numerical mismatches return GOAL_NOT_MET; malformed CLI/data errors remain ordinary errors. Cargo tests cover the supplied machinery and completed solution without requiring unfinished learner goals to pass.

## Boundaries

- Six course-authored vectors/pairs; default reported retrieval is on training pairs
- An unfamiliar interpolation is a narrow transfer check, not downstream validation
- Two-dimensional linear contrastive encoder and small tanh bottleneck
- The neighborhood browser tool uses designed vectors and a fixed decoder, not trained results

The numerical handoff and actual commands/results are in guidance/redesign/section-04.md. Browser tools are arithmetic illustrations and never execute Rust or certify completion. Source snippets carry exact data-source paths. The learning design follows the provided worked-example, retrieval and interactive research synthesis as a design inference, not evidence of measured learning gains in this course.
