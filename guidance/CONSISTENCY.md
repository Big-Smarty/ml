# Course terminology and code contract

This is the shared contract for the consistency revision requested by the learner. It applies to all 56 HTML lessons, examples, reference projects, starters, exercises, hints, answers, and glossary definitions. It supplements AUTHORING.md and NUMERICS.md. For this revision, each chapter is assigned to its own GPT-5.6 Sol High author; bounded research uses GPT-5.6 Luna High/Max. The lead integrates shared assets and reviews every chapter.

## Aim

The same concept keeps the same name, argument order, representation, and meaning until a new requirement justifies a change. Explain a necessary change before using it. Do not create a general framework or shared ML dependency to impose superficial uniformity. These remain independent learning checkpoints. Do not force trees, tokenizers, GPU devices, and differentiable models into one trait.

Read the preceding relevant chapter as well as the chapter being changed. Inspect all callers before renaming. Preserve numerical behavior, examples, datasets, CLI flags, checkpoint formats, and lesson scope. Where an existing correctness defect is discovered, report it and make a focused, tested correction; do not silently retune an experiment to hide it.

## First neuron: Chapters 1 and 2

Both chapters use the SAME `Neuron { weight: f64, bias: f64 }`, `Gradient { weight: f64, bias: f64 }`, and `TRAIN: [(f64, f64); 5]` fixture. A tuple always means `(input, target)` in that order. The model variable is `model`; data is `data`; one scalar input is `input`; its desired output is `target`.

The canonical signatures are:

```rust
impl Neuron {
    fn predict(&self, input: f64) -> f64;
    fn loss(&self, data: &[(f64, f64)]) -> Result<f64, &'static str>;
    fn numerical_gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str>;
    fn step(self, data: &[(f64, f64)], learning_rate: f64) -> Result<Self, &'static str>;
    fn train(self, data: &[(f64, f64)], steps: usize, learning_rate: f64) -> Result<Self, &'static str>;
}
```

Chapter 2 adds `fn gradient(&self, data: &[(f64, f64)]) -> Result<Gradient, &'static str>` and changes only which gradient `step` uses. Chapter 1 must introduce the small Rust struct, `impl`, `&self`, and `Result`/`?` before showing methods; these are programming organization, not unexplained ML concepts. `numerical_gradient` means central differences at h = 1e-5. Both derivatives are measured at the unchanged model. `step` returns the updated model. `train` loops `step`, takes the actual dataset explicitly, and does not hide TRAIN inside the algorithm.

`loss` is ordinary mean squared error: sum of `(prediction - target)^2`, divided by the number of examples, with NO factor 1/2. The initial loss is 9, the initial gradient is (-8,-2), and a 0.1 update gives (0.8,0.2). Loss, predict and train agree between the chapters; analytical versus numerical gradient is the learning objective.

Chapter 6's scalar regression model also uses `Neuron`, `predict`, and `loss` with these meanings. Its minibatch/regularization trainer may need extra arguments; show and explain them. Tiny standalone arithmetic exercises may isolate a primitive only when the lesson explicitly maps its inputs to the model fields. Prefer the same Neuron/method form in Chapters 1–2 exercises and starters where feasible, preserving learner-written solutions.

## Names and responsibilities

- A **parameter** is learned state (weight, bias, embedding, etc.); a **hyperparameter** is a chosen setting such as learning rate or batch size. Use `learning_rate`, not `rate`, `lr`, or `eta`, in the public teaching interface and ordinary Rust training code. Short names in mathematical equations, paper notation, shaders with external restrictions, or serialized formats must be mapped explicitly.
- `predict` produces a task prediction. Regression returns a numerical prediction. A classifier exposes `logit`/`logits`, `probability`/`probabilities`, and `predict` for its class decision where these operations exist; do not call raw scores probabilities. Classification's changed return meaning must be explained. Do not add unused methods solely to satisfy a naming table.
- `forward` computes a layer/network's outputs and, when needed, cached intermediate values for backpropagation. It is not an arbitrary replacement name for the scalar neuron's prediction. At the first transition explain that forward is the computation inside prediction. Keep `backward` for applying the chain rule to upstream gradients.
- `loss` on a model evaluates its documented scalar objective on explicitly supplied data/inputs and targets. Retain an existing Result/error boundary rather than changing errors into silent values. Avoid captured global datasets in model loss. Standalone loss primitives name their mathematics and input representation: `squared_error`, `mean_squared_error`, `binary_cross_entropy_from_logit`, `cross_entropy_from_logits`, etc. Their arguments put predictions/logits before targets. Do not rename a task-specific likelihood or ranking loss to a misleading MSE.
- `gradient` computes parameter derivatives of that same loss. `numerical_gradient` checks them with finite differences. A fused return of loss plus derivatives is spelled **`loss_and_gradient`**, not `loss_grad` or `loss_and_grad`. GPU variant: `loss_and_gradient_with_gpu`; backend variant: `loss_and_gradient_backend`. Explain the fusion when introduced: it shares intermediate computations, without changing what the loss or gradient means.
- `step` means one optimizer update when attached to a model/optimizer. `train` means repeated updates, `epochs` means dataset passes, and `steps` means optimizer updates. Token decoding, environment steps, and scheduler steps are other operations; identify the owning type and unit. Preserve explicit names such as `train_step` where needed to disambiguate.
- `fit` is appropriate for estimating preprocessing statistics or classical estimators (Scaler::fit, tree fitting). Explain that it estimates from training data. `transform` applies fitted preprocessing without refitting.
- Ordinary model methods borrow `&self` to inspect. Consuming `self -> Self` update methods in the early course remain deliberate. Later `&mut self` updates avoid copying large arrays; explain this ownership change. Gradient data is not a trained model: use a `Gradient` struct or documented arrays, rather than silently returning `Self` as a fake model.
- A scalar parameter is `weight`; multiple are `weights`; model bias is `bias`. `input` is a scalar/model input, `features` is a feature vector, `inputs` is a collection/tensor, `target` is one desired answer, `targets` are multiple answers. `label` remains legitimate for a discrete class ID or for APIs/languages where target is reserved (WGSL). Define that relationship. `error = prediction - target` is signed residual; it is distinct from nonnegative loss.
- Use `data`, `train_data`, `validation_data`, `test_data` consistently. **Example** is one supervised input/target pair; **sample** may describe a random draw; **batch** is the examples/tokens processed together. **Inference** uses learned parameters without updating them. Evaluation computes held-out metrics. Validation selects settings; test is the final untouched evaluation. Preserve source dataset names on disk.

## Shapes, reductions, and optimizations

In language-model formulas, use T for sequence/target positions, D for model width, V for vocabulary size, and τ for sampling or distillation temperature. Do not reuse T for temperature in a passage that also counts token positions; Rust spells the setting `temperature`.

State the input and output shapes next to the first public operation and every shape change. Default: contiguous row-major, activations [batch, features], dense weights [out_features, in_features], output = input × weights-transpose + bias. Existing different internal/kernel layouts may be retained when the lesson explicitly gives their indexing, transpose/conversion, and reason. Matrix multiplication is A[m,k] × B[k,n] = C[m,n]; loops can use row, col, inner. Do not describe [in,out] as [out,in]. Do not reorder checkpoint bytes for a cosmetic convention.

State exactly which elements/examples/tokens the loss averages over, and use the same reduction in its gradient. Name intentionally summed or per-example operations (`gradient_sum`, per-token loss). MSE uses no half factor; if a distinct objective needs half scaling, give it a distinct explicit name and explain. For cross-entropy use logits and stable log-sum-exp; do not change existing stable formulas while renaming. Regularized objective, unregularized data loss, and evaluation metric must be distinguished.

f64 is the early numerical reference; f32 is introduced for CPU/GPU performance, and tolerances replace bit equality. Keep scalar oracles and optimized versions clearly paired by suffix or module (`*_scalar`, `*_parallel`, GPU method). Existing established kernel names may stay if their roles are mapped. Do not change benchmark scope or claim newly measured hardware results from a CPU test.

## Required chapter revision

1. Read the whole lesson and all chapter code, including starter and exercise/solution. Find and fix inconsistencies throughout explanations, Rust blocks, hints, answers, outputs, source docs, and metadata; adding a note alone is not enough when code can be unified.
2. Add a concise early `<section id="continuity"><h2>Code and terminology carried forward</h2>...` after the lead/goal and before technical use. State the previous relevant model/API, what remains, and the exact justified changes here. Link `/conventions.html` and the relevant previous chapter. A small table of actual operations, data shapes, and reduction is useful; avoid generic boilerplate. Include only operations actually implemented. Chapters without a model (statistics/data/profiling/tokenization) say what earlier output they consume and which artifact they produce.
3. Rust blocks presented as implementation must match actual code or explicitly identify a short excerpt/context (for example, 'inside the training loop'). Do not quietly change receivers, argument order, return types, or reduction in a pedagogical simplification. Preserve purposeful broken debugging snippets, explicitly labeled as broken. For an exact reference-code excerpt, add `data-source="projects/chNN/src/main.rs"` (or the actual file) to its code tag; the consistency check verifies it against that source ignoring whitespace. Prefer this for core function definitions and reusable calls, without tagging intentionally partial or broken snippets. Source links target the defining file, including lib.rs when appropriate.
4. Chapter-owned glossary definitions must use these meanings and avoid redefining an earlier slug differently. Report duplicates to the lead instead of editing another chapter. Keep meaningful source citations; no new online research is needed for purely local renaming.
5. Test before and after with the existing checks. Run scoped fmt, strict Clippy, reference tests, starter launch/test, and exercise/solution tests. Learner-completed exercises are allowed to PASS and must never be restored to TODO. Do not run long training or GPU hardware concurrently. Report optional feature compilation needs for integration.
6. Write `guidance/consistency/chNN.md`: changes and interface mapping, loss/shape conventions, code/prose checks made, exact validation results, preserved learner changes, unresolved issues (or none). This is review evidence, not a lecture.

## Ownership and compatibility

Canonical glossary owners are: logit (04), data leakage (05), row-major order (09), checkpoint (11), teacher forcing (23), embedding and Recall@k (24). Later lessons link these definitions rather than adding competing copies. Also consolidate aliases: `logit-loss` links to `binary-cross-entropy` (04), `residual-path` links to `residual-connection` (21), and `moe-balancing-objective` links to `load-balancing-loss` (48). See `consistency/glossary-aliases.md` for the distinctions to preserve. Chapter-specific reductions and layouts still belong in each lesson.

Each author owns ONLY chapters/NN, projects/chNN, exercises/cpu/{exercises,solutions}/chNN_*.rs, and guidance/consistency/chNN.md. Root owns shared guidance, website shell, glossary integration, tools, and cross-chapter integration. No editing another chapter, no global fmt/build/verification while other authors write. Existing published APIs used by other projects require notifying root with old -> new signatures and every known caller. Root coordinates updates, including starters with path dependencies. Never leave deprecated duplicate teaching APIs merely to avoid updating known callers.

User work existed before this revision in projects/ch01/starter/src/main.rs and exercises/cpu/exercises/ch01_01.rs and ch02_01.rs. Its snapshot is /tmp/ml-before-consistency.tar.gz and its diff /tmp/ml-learner-before-consistency.diff. Preserve their completed code and experiments. Adapt names with care; do not erase the user's functions, restore TODOs, or redesign their experiment. Report any conflict to root.
