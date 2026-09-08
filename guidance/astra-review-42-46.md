# Independent Astra review: chapters 42–46

Reviewed and corrected 2026-09-08 by GPT-6 Astra High. This is substantive ownership of the completed chapter assets, building on the useful Sol High drafts; it is not a restatement of their earlier passing gates.

## Scope and research route

Read the full authoring contract, chapter template, numerical conventions, review contract, course manifest, Chapter 1 lecture and reference program. Independently read all five lessons, metadata, research records, Cargo manifests, reference programs, starters, and Rustlings exercises/solutions. Reviewed mathematical definitions, worked arithmetic, shape/layout contracts, implementation/prose agreement, promised algorithms, pedagogical sequence, citations, and limitations.

Commissioned two bounded, read-only primary-source verifications through collaboration tools:

- `verify_42_43`: GPT-5.6 Luna High. Checked online/tiled attention, GQA, actual SFT, LoRA, T=1 distillation; verified original FlashAttention, GQA, LoRA, distillation, and instruction-tuning sources. It independently ran the earlier references, confirmed measured outputs, and recommended clarifying constant teacher entropy and unmeasured student speed.
- `verify_44_46`: GPT-5.6 Luna High. Checked BM25/DPR, extractive retrieval, Q-learning, REINFORCE, DPO, and verifier feedback against primary papers. Found the chapter 45 worked-return error, missing reward/beta in the DPO derivation, an overclaim about validation of unused rows, and unchecked arithmetic addition. All were corrected by the Astra owner.

The per-chapter research records retain primary source URLs and now record the independent Astra/Luna route. Neither research agent authored lecture or implementation changes.

## Chapter 42: efficient attention

Confirmed that the CPU implementation visits every causal query/key pair and correctly merges running maximum, denominator, and value numerator. GQA uses contiguous groups with Q[token,query_head,lane] and K/V[token,kv_head,lane]. This is mathematically exact dense attention with reordered f32 arithmetic, not a GPU FlashAttention implementation.

Corrections: explicitly taught contiguous layouts; replaced an absolute-only test criterion with `2e-6 + 2e-6 * |reference|`; added a hand-computed increasing-maximum case, explicit four-query/two-KV mapping check, score overflow/nonfinite checks, and tile-zero coverage. Invalid CLI arguments now fail with usage. Benchmark prints architecture, OS, shape, precision, thread count, warmups/sample count, median and range. Both code and prose identify the measured boundary as complete allocating functions including validation and temporary storage; no allocation-free kernel timing is claimed.

Measured default: maximum absolute full/tiled error `0.00000006`; theoretical full score tensor 400 bytes. Optional benchmark on AMD Ryzen 9 9900X, Linux x86_64, one thread, f32, 128 tokens, 8 query heads, 2 KV heads, dimension 16, tile 16: 5 warmups and 7 samples. One measured full median was 1.190062 ms (1.183172–1.201253 ms), tiled median 0.906875 ms (0.896715–0.920425 ms), error `0.00000012`. These are local end-to-end function samples, not portable performance claims.

Limits: scalar CPU; no attention backward, GPU execution or trained-model quality measurement. The full oracle stores one row, not the theoretical quadratic tensor used in the memory example.

## Chapter 43: adaptation

Confirmed real Chapter 36 decoder pretraining before all-parameter SFT, output-projection LoRA with A[D,r]B[r,V] and simultaneous factor gradients, and teacher/student distillation with a smaller student. Only adapter factors mutate during LoRA. Weighted hard-label backwards equal the soft-target cross-entropy gradient at T=1.

Corrections: replaced probability-floor clipping in reported KL with direct stable log-softmax differences; a confidently wrong `[−1000,0]` student now reports the true approximately 1000 divergence against the opposite concentrated teacher. Added teacher-distribution shape/mass validation and a finite-difference check of the actual weighted gradient against direct soft cross-entropy. Explained low matrix rank, the constant teacher-entropy difference between KL and cross-entropy, old-factor updates, and implementation costs. LoRA still computes a full decoder backward on a cloned effective model, so its trainable parameter count is not presented as measured memory or speed savings. Student speed is explicitly implementation/hardware-dependent and unmeasured.

Measured default: SFT `1.3315 → 0.0147`; LoRA initial change 0, loss `1.3315 → 0.0016`, 32 trainable values against 784 base parameters; student mean KL `1.79547 → 0.01147`.

Limits: tiny synthetic pretraining/instruction sequences; no useful general language-model or held-out adaptation claim; only output LoRA; full-sequence loss rather than response masks; T=1 and V=8 backwards per transfer token. Only the existing local std-only ch36 dependency is used.

## Chapter 44: retrieval and RAG

Confirmed positive-smoothed BM25, a trained two-table averaged-word dual retriever, exhaustive dot-product search, source-labeled context, extraction from retrieved text, and abstention. The answer is not an LM or a canonical-answer lookup: changing retrieved text changes it without retraining.

Corrections: added direct-loss finite differences for both query and document embedding-table updates, with the old query/document state kept for both gradients. Strengthened exact-search Rustlings tests with a non-first winner and negative scores. Explained that all-zero BM25 scores for the no-overlap query yield only a tie-break winner, not lexical evidence. Clarified that unknown-content abstention is a limited fixture check: familiar words in an unsupported question can pass the heuristic gate. Extraction establishes textual provenance, not that the sentence answers the question. Added the known threshold ceiling and required calibration direction in a `ponytail:` comment.

Measured default: nearby held-out feature combinations hit@1 `3/3`; supported source `2/2`; coverage `2/3`; matching source labels `2/2`; the unknown-content unsupported query abstains `1/1`.

Limits: three facts and tiny training pairs; averaged embeddings and a fixed vocabulary; exact exhaustive search; extractive answering cannot synthesize; fixed score/margin thresholds are uncalibrated and do not guarantee support. The lesson explains where a real generative decoder would enter without pretending this program implements generation.

## Chapter 45: reinforcement learning

Confirmed epsilon-greedy sample means, tabular off-policy Q-learning, and actual sampled REINFORCE. REINFORCE computes its advantage from the previous baseline and changes both logits using old probabilities.

Corrections: the four-step worked discounted return is `0.800325`, not `0.802`. The terminal state is now absorbing and emits no repeated reward if called again. The shared target computation omits continuation on terminal transitions; a test passes a 99-valued continuation sentinel and checks it is ignored. Learned right-action values are compared with exact discounted path returns, not only greedy direction. Explained return-to-go, state-dependent action-independent baselines, and the importance of computing advantage before updating the baseline. Added the literal guided TODO marker to the starter.

Measured default: bandit estimates `[0.143,0.497,0.851]`; all four nonterminal states choose right; REINFORCE `[0.500,0.500] → [0.002,0.998]`.

Limits: one-dimensional chain (a five-cell grid), stationary simulated bandits, finite seeded interactions and constant learning rates; no convergence claim or multi-step policy-gradient credit assignment.

## Chapter 46: preference and reward training

Confirmed categorical DPO with frozen reference and simultaneous averaged pair gradients; stable softplus; actual parsed arithmetic reward and exact expected policy-gradient ascent.

Corrections: included `exp(reward/beta)` and explained beta in the KL-regularized objective; stated that finite separable preference fitting does not enforce a hard KL bound or retention outside comparisons. Stable log-softmax now computes `(logit−maximum)−logsumexp_shifted` so equal logits near 1e16 retain their log-two normalization. Validates every policy/reference row, including unused rows. Failed overflowing updates now preserve the entire old policy. Arithmetic parses use `checked_add` so extreme i64 inputs produce errors rather than debug panic/release wrap. Added pair-order invariance, frozen-reference, large-offset, extreme-margin, unused-shape, failed-update-preservation and arithmetic edge checks. Starter tests now require correct positive-margin and stable extreme-margin behavior; exercise tests cannot pass by returning a constant log two. Clarified that the starter enumerates the expectation of chapter 45's sampled policy-gradient update.

Measured default: DPO mean loss `0.6931 → 0.0718`; preferred responses become policy maxima; verifier policy `[0.0036,0.9928,0.0036]`.

Limits: categorical complete-response choices, not autoregressive preference fine-tuning; no response masking or length experiment; consistent authored pairs; exact three-action expectation does not test exploration, verifier gaming, or delayed credit.

## Exact validation performed

For every NN in 42,43,44,45,46:

- `cargo fmt --check --manifest-path projects/chNN/Cargo.toml` — PASS.
- `cargo clippy --offline --manifest-path projects/chNN/Cargo.toml --all-targets -- -D warnings` — PASS.
- `cargo test --offline --manifest-path projects/chNN/Cargo.toml` — PASS: 4,4,4,5,4 tests respectively, 21 total.
- `cargo run --offline --release --manifest-path projects/chNN/Cargo.toml` — PASS with useful measured outputs above.
- Same scoped fmt/clippy commands for `projects/chNN/starter/Cargo.toml` — PASS.
- `cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml` — PASS and useful earlier-knowledge computation.
- `cargo test --offline --manifest-path projects/chNN/starter/Cargo.toml` — expected nonzero exit, explicitly verified `not yet implemented` from its guided TODO.
- `rustfmt --edition 2021 --check exercises/cpu/{exercises,solutions}/chNN_01.rs` — PASS.
- `rustc --edition=2021 --test -D warnings exercises/cpu/solutions/chNN_01.rs -o /tmp/astra-NN-solutions` and run — PASS.
- Equivalent exercise compilation — PASS; running exercises — expected guided `not yet implemented` failure.

Also ran the chapter 42 optional release benchmark and repeated its full scoped gates after the last mapping assertion. Source HTML inspection confirmed no h1 shells, unique ids, existing local code/chapter links, and known glossary targets. Each lesson exceeds 1,500 meaningful words (approximately 1,758 / 1,854 / 1,789 / 2,078 / 2,077 respectively); glossary explanations are 60–120 words.

No global builder, global verifier, shared assets, or other authors' code was edited or run. Integrated browser/rendered-page verification and final course-wide gates remain the lead's responsibility; this record does not claim those checks were performed here.

## Files owned

The five chapters' `lesson.html`, `meta.json`, and `research.md`; five reference `projects/chNN/src/main.rs`; chapter 45/46 starter source markers/tests; chapter 44/46 Rustlings exercise and solution tests; this review record. Scoped formatting also normalized already-owned chapter source formatting. No new runtime dependencies or shared abstractions were introduced.
