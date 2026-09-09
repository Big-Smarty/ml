# Chapter 21: redesign research and implementation record

Author route: one GPT-6 Astra High section owner, 2026-09-09. Read the complete previous lesson, metadata, research notes, reference and starter sources before redesign. The active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract replaces historical Rustlings and broken-starter requirements. Reference projects remain unchanged; tested numerical algorithms are adapted into labs/s04-deep-learning with learner-selected cores and separate explained solutions.

## Evidence and source claims

The following primary sources were previously verified in the course's 2026-09-08 research and independent review. This redesign reuses those bounded claims; it does not claim a new literature search or reproduction of the papers' full experiments.

- Deep Residual Learning for Image Recognition — https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
  Original residual-network paper.
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift — https://proceedings.mlr.press/v37/ioffe15.html
  Primary batch-normalization paper.
- Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis — https://doi.org/10.1109/ICDAR.2003.1227801
  Primary study of CNN image distortions on MNIST.
- AutoAugment: Learning Augmentation Strategies From Data — https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html
  Primary learned augmentation-policy study.

## Active learning route

- Session 1: Implement per-image mean/variance scaling and verify shifted and constant arrays.
- Session 2: Differentiate normalization and trace first-block credit through both blocks.
- Session 3: Implement zero-padded translation and compare canonical/five-shift policies.

Every session has worked values, named Rust work, a checkpoint or controlled experiment, optional explanation and a changed-input transfer task. Baseline success demonstrates a functioning earlier model, not the completed learning goal. The chapter's --check calls the selected learner core. Intentional numerical mismatches return GOAL_NOT_MET; malformed CLI/data errors remain ordinary errors. Cargo tests cover the supplied machinery and completed solution without requiring unfinished learner goals to pass.

## Boundaries

- Per-image normalization without learned affine parameters, not batch normalization
- Two single-channel blocks and a two-class head
- Four validation arrays remain perturbations of the same source patterns
- Tiny baseline can match or beat the more elaborate model; no universal normalization/augmentation advantage

The numerical handoff and actual commands/results are in guidance/redesign/section-04.md. Browser tools are arithmetic illustrations and never execute Rust or certify completion. Source snippets carry exact data-source paths. The learning design follows the provided worked-example, retrieval and interactive research synthesis as a design inference, not evidence of measured learning gains in this course.
