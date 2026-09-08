# Final cross-chapter consistency review: Chapters 42–45

Scope: current lessons, metadata, research notes, reference projects, starters, Rustlings exercises/solutions, and the relevant Chapter 24/35/36/40 interfaces. Focused checks ran with `cargo test --offline --manifest-path projects/ch42/Cargo.toml` through `projects/ch45/Cargo.toml`: 4, 4, 4, and 5 reference tests passed.

## Findings

1. **Chapter 43 adaptation table overstates/generalizes private APIs** — [chapters/43/lesson.html:6](/home/bigsmarty/Projects/ml/chapters/43/lesson.html:6) describes `sft` as accepting generic `[T]` pairs and lists `Lora::step`, `distill_gradient`, and `mean_teacher_kl` as bare scalar/gradient returns. The actual `sft` helper is fixed to `[usize; 3]` ([projects/ch43/src/main.rs:35](/home/bigsmarty/Projects/ml/projects/ch43/src/main.rs:35)); the other three functions return `Result` ([projects/ch43/src/main.rs:117](/home/bigsmarty/Projects/ml/projects/ch43/src/main.rs:117), [projects/ch43/src/main.rs:172](/home/bigsmarty/Projects/ml/projects/ch43/src/main.rs:172), [projects/ch43/src/main.rs:216](/home/bigsmarty/Projects/ml/projects/ch43/src/main.rs:216)). Fix the table to say fixture `[3]` (or explicitly “generic decoder slice, fixture helper `[3]`”) and preserve the `Result<...>` boundaries.

2. **Chapter 45 claims actual helper reuse that is only copied logic** — [chapters/45/lesson.html:89](/home/bigsmarty/Projects/ml/chapters/45/lesson.html:89) calls the starter a carried-forward Chapter 44 retriever, and [guidance/consistency/ch45.md:57](/home/bigsmarty/Projects/ml/guidance/consistency/ch45.md:57) says it “truly reuses” Chapter 44 helpers. `projects/ch45/starter/src/main.rs:3-13` duplicates `words`/`overlap`; its Cargo manifest has no Chapter 44 dependency, and its fixtures/output differ from the Chapter 44 starter. Say it “copies the same helper logic” or add a real shared dependency; the minimal fix is the wording change.

3. **Stale Chapter 43 review-record claim** — [guidance/consistency/ch43.md:50](/home/bigsmarty/Projects/ml/guidance/consistency/ch43.md:50) says the starter’s `low_rank_delta` is called by `Lora::delta`, but the starter contains no `Lora` type or `delta` method; only the reference does ([projects/ch43/src/main.rs:76](/home/bigsmarty/Projects/ml/projects/ch43/src/main.rs:76)). Reword this as “matches the unscaled helper used by the reference’s `Lora::delta`.”

## No further findings

The score/probability/logit-gradient distinctions, `T`/`τ` usage, effective-decoder LoRA chain rule and frozen-base behavior, Q-value/return definitions, terminal bootstrap handling, and `environment_step` versus `q_update` ownership were consistent across the reviewed assets.

Root resolution: the Chapter 43 table now states fixed T=3 for the SFT helper and preserves Result boundaries. Both review records and the Chapter 45 lesson explicitly describe matching or copied helper implementations in independent projects. No new dependency or numerical change was needed.
