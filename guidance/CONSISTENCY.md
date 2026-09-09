# V2 curriculum consistency contract

This is the active cross-course consistency contract for the September 2026 learning redesign. It applies to all 56 lessons, nine cumulative lab packages, section reports, browser interactions, explanations, checks, and solutions.

The active sources of authority are:

1. [AUTHORING.md](AUTHORING.md) for the learning, content, lab, metadata, interaction, and handoff contract.
2. [ASSIGNMENTS.md](ASSIGNMENTS.md) for section ownership, chapter ranges, lab packages, project arcs, and the 18-interaction allocation.
3. [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md) for evidence, counterevidence, transfer limits, and assessment validity.
4. [NUMERICS.md](NUMERICS.md) for numerical practice when it does not conflict with the v2 authoring contract.
5. The owner-authored `guidance/redesign/section-NN.md` matrices for chapter-specific interfaces, sessions, outcomes, checks, and cross-section handoffs.
6. [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md) for observed v2 validation results and remaining gates.

Pre-v2 consistency, research, review, and validation records are preserved in [archive/README.md](archive/README.md). They are historical evidence and are not active instructions.

## Purpose

The same concept should keep the same name, shape, reduction, unit, and meaning while it is reused. When a later chapter needs a different representation or ownership model, explain the change before using it and verify the boundary. Consistency serves learner understanding and executable continuity; it does not require one universal API.

Do not force scalar models, trees, tokenizers, GPU devices, retrieval systems, and differentiable networks into a shared trait. Reuse an interface only when learner work or a section project actually crosses that boundary.

## Vocabulary and operation names

- A **parameter** is learned state. A **hyperparameter** is selected by the experimenter. Public teaching interfaces use descriptive names such as `learning_rate`; abbreviated paper, shader, or serialization names are mapped explicitly.
- A regression **prediction** is a numerical task output. A classification **logit** is an unnormalized score, a **probability** is a normalized value with a stated interpretation, and a class prediction is the resulting decision. Do not call raw scores probabilities.
- `forward` computes layer or network outputs and any intermediates needed later. `backward` propagates upstream derivatives and produces parameter or input derivatives. Explain their relationship when a chapter moves from a task-level `predict` method to layered computation.
- `loss` is the documented scalar training objective on explicit inputs and targets. Name standalone objectives precisely: `mean_squared_error`, `binary_cross_entropy_from_logits`, `cross_entropy_from_logits`, ranking loss, contrastive loss, and so on.
- `gradient` means derivatives of the stated objective. `numerical_gradient` or `finite_difference_check` is a checking procedure. A fused calculation uses a clear name such as `loss_and_gradient` and states which intermediate work is shared.
- `step` or `train_step` means one optimizer update. `steps` count updates; `epochs` count passes through a dataset. Decode steps, environment steps, scheduler steps, and pipeline stages name their owner and unit.
- `fit` estimates state from training data. `transform` applies already fitted state. Preprocessing never silently refits on validation or test data.
- `input` names one model input, `features` a feature vector, `inputs` a collection or tensor, `target` the desired response, and `label` a discrete class identifier where appropriate. Define any different external API terminology.
- A signed residual or prediction error is distinct from a nonnegative loss. Training, inference, validation, and final evaluation retain their separate meanings.
- An **example** is one supervised unit, a **sample** may be a random draw, and a **batch** is the collection processed together. State the sampling unit for uncertainty calculations.

Chapter prose, browser labels, Rust symbols, test output, and section reports use the same concept names unless a local technical constraint is explicitly mapped.

## Shapes, layouts, reductions, and units

State input and output shapes beside the first operation and at every shape change. Show a small concrete case before a symbolic generalization.

The default dense teaching convention is row-major activations `[batch, features]`, dense weights `[out_features, in_features]`, and `output = input × weightsᵀ + bias`. A chapter may retain a different established kernel, checkpoint, or framework layout when it gives the indexing rule, transpose or conversion, and reason. Never relabel stored bytes to create cosmetic uniformity.

Matrix multiplication uses `A[m, k] × B[k, n] = C[m, n]` unless a chapter explicitly defines a different notation. Language-model prose normally uses `T` for sequence positions, `D` for model width, `V` for vocabulary size, and `τ` for temperature. Avoid reusing one symbol for two meanings in the same explanation.

Every objective states whether it sums or averages over examples, tokens, classes, spatial positions, experts, or devices. Its gradient uses the same reduction. Distinguish:

- data loss from a regularized objective;
- training objective from evaluation metric;
- per-example values from batch reductions;
- attempted expert routing from admitted load;
- modeled bytes or operations from measured runtime;
- active parameters from total parameters, FLOPs, latency, and memory traffic.

Show units for durations, throughput, memory, probabilities, logits, rates, and hardware measurements. State deterministic seeds, tolerances, and relevant environment metadata beside results.

Early numerical references may use `f64`; CPU/GPU paths commonly use `f32` or supported lower precision. Explain the precision transition and compare within justified tolerances rather than bit equality.

## Learning-path consistency

Every chapter can span several 30–45-minute sessions. Session structure and stop conditions come from its section matrix. Across the course, preserve this recognizable progression without turning it into repetitive prose:

1. retrieve a needed dependency;
2. run and inspect a meaningful working baseline;
3. connect concrete values, a visual or trace, notation, and Rust;
4. predict one interpretable result;
5. implement a substantial algorithmic seam;
6. verify it with readable local evidence;
7. diagnose or vary one meaningful condition;
8. explain an inference and limitation;
9. reuse the concept later with less guidance.

Prediction, answers, hints, continuous view, and navigation remain freely available. Browser progress is explicit self-report and never implies mastery.

The active path uses the nine v2 lab packages in [ASSIGNMENTS.md](ASSIGNMENTS.md). Do not reintroduce Rustlings, intentionally broken startup, missing-expression drills, or the old per-chapter starter/reference workflow. A learner baseline runs and teaches something before edits. Learning-goal checks may report `GOAL_NOT_MET:` only for an expected unfinished algorithm, and `GOAL_REVIEW_REQUIRED:` only for a documented manual criterion, as specified in [AUTHORING.md](AUTHORING.md).

## Cross-chapter continuity

Section owners read all chapters in their range and relevant predecessor outputs. The nine `guidance/redesign/section-NN.md` reports collectively form the authoritative 56-row design matrix for:

- topic-to-step coverage;
- session goals and ordering;
- learner-owned functions or shader regions;
- supplied plumbing;
- commands and expected evidence;
- unfamiliar transfer cases;
- interactive fixture values;
- output artifacts and downstream consumers;
- tests run and unresolved boundaries.

When another section consumes an artifact, record its exact format, shape, units, seed/configuration, error behavior, and ownership. Ask the lead to coordinate any cross-section API or shared-data change before editing another owner's files.

Keep CPU or simple reference oracles beside optimized paths. GPU conceptual work, actual dispatch correctness, and performance measurements are separate evidence states. A browser explanation or CPU fallback does not satisfy a real-hardware milestone.

## Prose, code, and source alignment

- A code block presented as executable implementation matches the named v2 lab file or is labeled as pseudocode, an excerpt, or intentionally faulty diagnosis code.
- Every formula defines its symbols, shapes, units, and reduction.
- Every claimed observation identifies whether it came from a worked fixture, deterministic local run, stochastic experiment, analytical model, or external source.
- Browser calculations are labeled illustrations and link to the corresponding local Rust seam and check.
- Glossary definitions preserve one meaning per slug. Later chapters link to the established definition and add local nuance in prose.
- Primary technical sources support algorithmic and quantitative claims. [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md) supports pedagogical choices and states their limits.
- Speed, quality, scale, hardware, and generalization claims remain bounded to the actual model, data, device, and measurement.

## Required consistency evidence

Before section handoff, the owner records in `guidance/redesign/section-NN.md`:

- all assigned `course.json` topics and their lesson-step IDs;
- every chapter's sessions and observable stop conditions;
- lab entry, learner seam, supplied infrastructure, solution, and transfer check;
- shapes, layouts, reductions, units, and cross-chapter artifacts;
- exact validation commands and observed outcomes;
- browser-interaction fixture parity and accessibility evidence where assigned;
- source verification and claim limits;
- unresolved integration or hardware work.

Review requirements live in [REVIEW.md](REVIEW.md). Only observed integrated results belong in [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md). Do not copy historical pass claims into v2 status.
