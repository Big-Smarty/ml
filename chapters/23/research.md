# Chapter 23: redesign research and implementation record

Author route: one GPT-6 Astra High section owner, 2026-09-09. Read the complete previous lesson, metadata, research notes, reference and starter sources before redesign. The active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract replaces historical Rustlings and broken-starter requirements. Reference projects remain unchanged; tested numerical algorithms are adapted into labs/s04-deep-learning with learner-selected cores and separate explained solutions.

## Evidence and source claims

The following primary sources were previously verified in the course's 2026-09-08 research and independent review. This redesign reuses those bounded claims; it does not claim a new literature search or reproduction of the papers' full experiments.

- Finding Structure in Time — https://doi.org/10.1207/S15516709COG1402_1
  Primary simple recurrent-network study.
- Backpropagation through time: what it does and how to do it — https://doi.org/10.1109/5.58337
  Primary practical BPTT account.
- Long Short-Term Memory — https://doi.org/10.1162/neco.1997.9.8.1735
  Original LSTM paper.
- Out-of-sample tests of forecasting accuracy: an analysis and review — https://doi.org/10.1016/S0169-2070(00)00065-0
  Primary review of fixed and rolling forecasting evaluations.
- A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction — https://doi.org/10.1016/j.csda.2017.11.003
  Study of conditions under which ordinary CV is or is not appropriate for autoregression.
- Learning to Forget: Continual Prediction with LSTM — https://doi.org/10.1162/089976600300015015
  Primary source for the adaptive forget gate in the modern LSTM variant.

## Active learning route

- Session 1: Implement recurrent forward and explain its signed-input cache.
- Session 2: Accumulate all shared RNN parameter derivatives through time.
- Session 3: Carry hidden and cell derivatives through a modern LSTM.
- Session 4: Implement generated-prefix LSTM rollout and compare horizon-specific errors.

Every session has worked values, named Rust work, a checkpoint or controlled experiment, optional explanation and a changed-input transfer task. Baseline success demonstrates a functioning earlier model, not the completed learning goal. The chapter's --check calls the selected learner core. Intentional numerical mismatches return GOAL_NOT_MET; malformed CLI/data errors remain ordinary errors. Cargo tests cover the supplied machinery and completed solution without requiring unfinished learner goals to pass.

## Boundaries

- One hidden scalar per model and a deterministic course-authored signal
- One chronological origin; no uncertainty or real-world forecasting claim
- Teacher-forced and free-running numbers use different input protocols and target ranges
- Generated-prefix replay is O(horizon²), bounded to40 by CLI; a stateful API is the long-horizon upgrade

The numerical handoff and actual commands/results are in guidance/redesign/section-04.md. Browser tools are arithmetic illustrations and never execute Rust or certify completion. Source snippets carry exact data-source paths. The learning design follows the provided worked-example, retrieval and interactive research synthesis as a design inference, not evidence of measured learning gains in this course.
