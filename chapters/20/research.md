# Chapter 20: redesign research and implementation record

Author route: one GPT-6 Astra High section owner, 2026-09-09. Read the complete previous lesson, metadata, research notes, reference and starter sources before redesign. The active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract replaces historical Rustlings and broken-starter requirements. Reference projects remain unchanged; tested numerical algorithms are adapted into labs/s04-deep-learning with learner-selected cores and separate explained solutions.

## Evidence and source claims

The following primary sources were previously verified in the course's 2026-09-08 research and independent review. This redesign reuses those bounded claims; it does not claim a new literature search or reproduction of the papers' full experiments.

- Learning representations by back-propagating errors — https://doi.org/10.1038/323533a0
  Original concise account of learning hidden representations by backpropagation.
- Neocognitron — https://doi.org/10.1007/BF00344251
  Early hierarchical local-feature and position-tolerance architecture.
- Gradient-Based Learning Applied to Document Recognition — https://doi.org/10.1109/5.726791
  Primary LeNet account of local receptive fields, shared weights, subsampling, and digit recognition.
- The MNIST Database of Handwritten Digits — https://yann.lecun.org/exdb/mnist/
  Official dataset description and IDX format source.

## Active learning route

- Session 1: Compute a complete padded response map and explain a rectangular boundary case.
- Session 2: Implement max pooling and preserve the correct winner indices on odd dimensions.
- Session 3: Accumulate all nine shared derivatives and verify complete training.
- Session 4: Run a controlled reduction comparison and trace a new image.

Every session has worked values, named Rust work, a checkpoint or controlled experiment, optional explanation and a changed-input transfer task. Baseline success demonstrates a functioning earlier model, not the completed learning goal. The chapter's --check calls the selected learner core. Intentional numerical mismatches return GOAL_NOT_MET; malformed CLI/data errors remain ordinary errors. Cargo tests cover the supplied machinery and completed solution without requiring unfinished learner goals to pass.

## Boundaries

- Two filters and sample-at-a-time scalar CPU SGD
- Procedural seven-segment training and validation share a source; no population accuracy claim
- Stride-one padding-one convolution and floor-sized2×2 pooling are fixed in the local classifier
- MNIST requires explicit local paths and limits fitting/validation to2000 images each; no new MNIST result claimed

The numerical handoff and actual commands/results are in guidance/redesign/section-04.md. Browser tools are arithmetic illustrations and never execute Rust or certify completion. Source snippets carry exact data-source paths. The learning design follows the provided worked-example, retrieval and interactive research synthesis as a design inference, not evidence of measured learning gains in this course.
