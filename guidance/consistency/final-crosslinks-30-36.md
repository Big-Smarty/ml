# Final cross-chapter audit: Chapters 30–36

Read-only audit of the frozen lessons, metadata, research notes, reference code, starters, exercises, solutions, and relevant downstream callers. No chapter or shared source file was changed.

## Actionable findings

- **[P2] Chapter 36 claims a checkpoint format that does not exist in Chapter 36.** [`chapters/36/lesson.html:8`](/home/bigsmarty/Projects/ml/chapters/36/lesson.html:8) says “Existing Chapter 36 checkpoints must keep” the decoder bytes, spans, and names, but [`projects/ch36/src/lib.rs:329`](/home/bigsmarty/Projects/ml/projects/ch36/src/lib.rs:329) only exposes parameters and spans; checkpoint I/O begins in Chapter 38 (`Trainer::save`/`load`, [`projects/ch38/src/lib.rs:199`](/home/bigsmarty/Projects/ml/projects/ch38/src/lib.rs:199)). Reword this as a downstream Chapter 38 checkpoint or serialized consumer contract so learners do not infer that Chapter 36 already writes checkpoints.

- **[P2] Chapter 36 presents `trace()` as the cached-inference handoff, but Chapter 40 does not use it.** [`chapters/36/lesson.html:37`](/home/bigsmarty/Projects/ml/chapters/36/lesson.html:37) says `trace()` exposes Q/K/V and probabilities “for the cached-inference chapter.” Chapter 40's [`CachedDecoder::new`] reads `parameter_spans()` and its own incremental `step` maintains per-layer K/V caches ([`projects/ch40/src/main.rs:11`](/home/bigsmarty/Projects/ml/projects/ch40/src/main.rs:11), [`projects/ch40/src/main.rs:18`](/home/bigsmarty/Projects/ml/projects/ch40/src/main.rs:18)); no caller of `Decoder::trace` exists. Describe `trace()` as a Chapter 36 diagnostic helper, and say Chapter 40 reuses the decoder plus spans while implementing its own cache path.

## Checks

The scoped consistency checker reports no Chapters 30–36 continuity, convention-link, source-excerpt, glossary-ownership, or API-name finding. Manual comparison found no additional unexplained signature changes, probability/parameter-weight confusion, `[out,in]` versus `[in,out]` transpose or gradient error, stale Chapter 30–36 downstream caller, or false `data-source` excerpt. The corrected Chapter 35 projection-gradient paragraph and Chapter 36 starter carry-forward now agree with their implementations.

Root resolved both final crosslink findings: checkpoint I/O is explicitly introduced in Chapter 38; trace() is a diagnostic helper, while Chapter 40 consumes the model and parameter spans for its own incremental cache. No code changed.
