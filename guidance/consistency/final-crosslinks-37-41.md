# Final cross-chapter audit: Chapters 37–41

Read-only audit of the frozen lessons, metadata, research notes, reference code, starters, solutions, and relevant Chapter 34/36 callers. No chapter or shared source file was changed.

## Actionable findings

- **[P2] A step-zero Chapter 38 checkpoint cannot enforce the documented training-data identity check.** [`projects/ch38/src/lib.rs:57`](/home/bigsmarty/Projects/ml/projects/ch38/src/lib.rs:57) initializes `data_fingerprint` to `0`, and [`projects/ch38/src/lib.rs:109`](/home/bigsmarty/Projects/ml/projects/ch38/src/lib.rs:109) rejects changed data only when the saved fingerprint is nonzero. A checkpoint saved before its first update therefore accepts any data on the first resumed update, despite [`chapters/38/lesson.html:50`](/home/bigsmarty/Projects/ml/chapters/38/lesson.html:50) and [`chapters/39/lesson.html:69`](/home/bigsmarty/Projects/ml/chapters/39/lesson.html:69) describing the fingerprint check as generally binding a resumed run. Either reject step-zero saves for exact-resume use or explicitly qualify the prose and checkpoint state as unbound until the first update.

- **[P3] Chapter 38’s final review record contains a stale post-revision starter count.** [`guidance/consistency/ch38.md:42`](/home/bigsmarty/Projects/ml/guidance/consistency/ch38.md:42) says the post-revision starter reported 10,960 parameters, while the frozen starter uses context 8 and therefore reports 10,832 ([`projects/ch38/starter/src/main.rs:22`](/home/bigsmarty/Projects/ml/projects/ch38/starter/src/main.rs:22)); the same report acknowledges the 10,832 integration at [`guidance/consistency/ch38.md:53`](/home/bigsmarty/Projects/ml/guidance/consistency/ch38.md:53). Correct the review evidence so final validation does not contradict the lesson and code.

## Checks

Scoped source-excerpt checks pass for Chapters 37–41. Manual comparison found no additional material false cross-chapter claim, unexplained public signature, `[out,in]`/`[in,out]` layout or gradient-math mismatch, artifact/provenance handoff error, logits/probability/step semantic error, stale lesson source excerpt, or unresolved Chapter 34/36 caller migration.

Root resolved the final audit findings: prose now distinguishes unbound step-zero checkpoints from nonzero bound training fingerprints, and the final Chapter38 starter count is 10,832. Model arithmetic, checkpoint format and runtime behavior are unchanged.
