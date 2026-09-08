# Chapter 10 research

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High primary-source research pass. The official MNIST page verified 60,000/10,000 counts, 28×28 unsigned pixels, big-endian IDX headers, and magic values 2051/2049. PyTorch’s loss documentation was used only to cross-check logits/cross-entropy conventions. The generated IDX fixture, parser, optimizer, prose, and tests are original. No MNIST files were downloaded or redistributed.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (08–14). Rearranged cross-entropy so the common maximum cancels the target score before adding the small log-sum-exp term; ten logits at 1e16 now retain ln 10. Added truncation, class-range, and count-mismatch checks and stronger softmax/log-sum-exp learner checks. No MNIST download or training was repeated during this review.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
