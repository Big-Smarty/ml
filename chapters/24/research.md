# Chapter 24: redesign research and implementation record

Author route: one GPT-6 Astra High section owner, 2026-09-09. Read the complete previous lesson, metadata, research notes, reference and starter sources before redesign. The active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract replaces historical Rustlings and broken-starter requirements. Reference projects remain unchanged; tested numerical algorithms are adapted into labs/s04-deep-learning with learner-selected cores and separate explained solutions.

## Evidence and source claims

The following primary sources were previously verified in the course's 2026-09-08 research and independent review. This redesign reuses those bounded claims; it does not claim a new literature search or reproduction of the papers' full experiments.

- Matrix Factorization Techniques for Recommender Systems — https://doi.org/10.1109/MC.2009.263
  Primary overview of latent-factor recommendation.
- Collaborative Filtering for Implicit Feedback Datasets — https://doi.org/10.1109/ICDM.2008.22
  Primary weighted factorization treatment of implicit feedback and confidence.
- BPR: Bayesian Personalized Ranking from Implicit Feedback — https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf
  Original BPR objective and LearnBPR algorithm.
- Cumulated Gain-based Evaluation of IR Techniques — https://doi.org/10.1145/582415.582418
  Primary source for discounted and normalized cumulative gain.

## Active learning route

- Session 1: Implement simultaneous user/positive/negative updates and verify six derivatives.
- Session 2: Implement binary nDCG with general multi-relevant denominators.
- Session 3: Compare an explicit negative policy and complete a controlled experiment across the section.

Every session has worked values, named Rust work, a checkpoint or controlled experiment, optional explanation and a changed-input transfer task. Baseline success demonstrates a functioning earlier model, not the completed learning goal. The chapter's --check calls the selected learner core. Intentional numerical mismatches return GOAL_NOT_MET; malformed CLI/data errors remain ordinary errors. Cargo tests cover the supplied machinery and completed solution without requiring unfinished learner goals to pass.

## Boundaries

- Four users,six items,two factors and explicit negatives
- Four candidates and one held-out positive per query, not full-catalog evaluation
- No biases, cold-start encoder, approximate retrieval or online outcome study
- The fixture does not estimate exposure effects, fairness, diversity or user satisfaction

The numerical handoff and actual commands/results are in guidance/redesign/section-04.md. Browser tools are arithmetic illustrations and never execute Rust or certify completion. Source snippets carry exact data-source paths. The learning design follows the provided worked-example, retrieval and interactive research synthesis as a design inference, not evidence of measured learning gains in this course.
