# Chapter 24 research notes

Route: chapter authored by the assigned high-reasoning author after bounded source verification by `gpt-5.6-luna` at high reasoning. Verification date: 2026-09-08. Interaction data and implementation are course-authored.

- Koren, Bell, and Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer 42(8) (2009), https://doi.org/10.1109/MC.2009.263. Supports shared latent user/item vectors and dot-product preference scores.
- Hu, Koren, and Volinsky, “Collaborative Filtering for Implicit Feedback Datasets,” ICDM 2008, https://doi.org/10.1109/ICDM.2008.22. Supports distinguishing implicit preference from confidence and using a weighted factor model.
- Rendle et al., “BPR: Bayesian Personalized Ranking from Implicit Feedback,” UAI 2009, https://www.auai.org/uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf. Supports user-positive-negative triples, a pairwise ranking objective, and stochastic optimization.
- Järvelin and Kekäläinen, “Cumulated Gain-based Evaluation of IR Techniques,” TOIS 20(4) (2002), https://doi.org/10.1145/582415.582418. Supports cumulative gain, logarithmic rank discounting, and normalization by an ideal order. This is an information-retrieval source applied to recommendation ranking.

The fixture supplies explicit negatives. Evaluation ranks one held-out positive against three explicit negatives, not the full six-item catalog. This avoids silently treating every unobserved interaction as negative, but it also narrows the claim.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 20–24; the researcher supplied checks and source findings, while Astra implemented the corrections.

Verified shared-factor pairwise updates, stable softplus, held-out interaction exclusion, deterministic candidate ranking and hand metrics. Clarified one-based nDCG rank in the glossary to agree with zero-based r+2 in the lesson/code. No reference algorithm change was needed; the explicit-negative four-item candidate restriction remains.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
