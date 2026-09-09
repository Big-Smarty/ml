# Chapter 10 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. The official MNIST page verified 60,000/10,000 counts, 28×28 unsigned pixels, big-endian IDX headers, and magic values 2051/2049. PyTorch’s loss documentation was used only to cross-check logits/cross-entropy conventions. The generated IDX fixture, parser, optimizer, prose, and tests are original. No MNIST files were downloaded or redistributed.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Rearranged cross-entropy so the common maximum cancels the target score before adding the small log-sum-exp term; ten logits at 1e16 now retain ln 10. Added truncation, class-range, and count-mismatch checks and stronger softmax/log-sum-exp learner checks. No MNIST download or training was repeated during this review.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.


## Section redesign — 2026-09-09

One GPT-6 Astra High owner authored Chapters 07–11 as a connected section. Read original lessons, metadata, research and both project variants before replacement. The existing source verification above grounds the mathematical claims; no new claim of MNIST performance is made. The new lab reuses the reference arithmetic and supplied parsing/checkpoint mechanisms, with an append-only scalar tape replacing Rc/RefCell infrastructure. Core exercises now call learner functions through explicit goal checks; normal tests validate functioning baselines and the separate completed solutions.

The route uses the supplied foundations audit, learning-science synthesis and interactive design research: runnable baseline, collocated arithmetic and code, progressive help, a meaningful implementation and changed-input evidence. These are design inferences, not evidence that this complete course format is experimentally proven superior. Session times are planning estimates, not measured learner times. See `guidance/redesign/section-02.md` for topic/action/check mapping and actual validation.
