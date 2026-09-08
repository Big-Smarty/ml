# Chapter 36 caller migration

Root updated every known downstream call after the Chapter 36 author applied the canonical `loss_and_gradient` and `loss_and_gradient_with_gpu` names. Parameter spans, layouts, arithmetic and checkpoint formats were untouched.

Changed files:

- `projects/ch38/src/lib.rs`
- `projects/ch39/src/main.rs`
- `projects/ch41/src/main.rs`
- `projects/ch43/src/main.rs`
- `projects/ch43/starter/src/main.rs`
- `chapters/43/lesson.html`

The later chapter authors validated these callers in their scoped passes. Root integrated the CPU checks and the optional GPU checks recorded in hardware.json.

The coordinated Chapter38 public schedule fields are now peak_learning_rate/min_learning_rate. No external direct field caller was found; Chapter39 names them in prose. Serialization order and the deterministic checkpoint digest are unchanged. Root revalidated38/39 CPU paths,40 cached inference, and tiny39 GPU save/load/resume after these changes.
