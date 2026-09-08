# Chapter 42 research notes

Route: Sol High author with bounded GPT-5.6 Luna High research. Primary sources verified 2026-09-08.

- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): running maximum and shifted denominator recurrence; stable bounds on the denominator.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135): exact tiled attention combines per-block maximum, denominator, and output numerator, reducing HBM traffic and avoiding quadratic intermediates. Dense arithmetic stays quadratic.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product equation and O(n²d) dense self-attention complexity.
- Ainslie et al., [GQA](https://aclanthology.org/2023.emnlp-main.298.pdf): each KV head serves a group of query heads; MHA and MQA are endpoint cases. Decoder-only implications are presented as mechanism, not copied empirical quality.

The project compares against its own full two-pass causal oracle. Tiling changes f32 accumulation order, so the acceptance check uses tolerance rather than bitwise equality.

## Independent Astra implementation review — 2026-09-08

Route: GPT-6 Astra High owner/reviewer, with fresh bounded GPT-5.6 Luna High primary-source verification (`verify_42_43` for 42–43; `verify_44_46` for 44–46). Earlier Sol High drafts were retained where correct; the Astra owner independently read and corrected all lesson, metadata, reference, starter, and exercise assets.

Verified online maximum/denominator/value-numerator recurrence, exact causal coverage, contiguous GQA grouping, MHA/MQA endpoints, and quadratic dense arithmetic. Added a hand-computed increasing-maximum case, nonfinite-score rejection, mixed tolerance, tensor layout, explicit end-to-end timing boundary and range. No GPU performance is claimed.

The Luna High verification checked the original papers and official source URLs listed above. Its concrete findings were integrated by the Astra owner; passing prior author gates was not treated as independent proof. Scoped validation and remaining limits are recorded in `guidance/astra-review-42-46.md`.
