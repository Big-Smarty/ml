# Read-only consistency audit: Chapters 29–56

Scope: lessons, references, starters, exercises, and cross-crate callers. No chapter files were changed.

## Findings

### High: shared naming contract conflicts with the implemented API

`guidance/CONSISTENCY.md:35,39` requires `learning_rate` in ordinary teaching code and spells the fused loss/gradient API `loss_and_gradient` (with `loss_and_gradient_with_gpu` and `loss_and_gradient_backend` variants). The scoped code instead uses `loss_and_grad` throughout the Chapter 36 family and uses `rate` for learning-rate parameters in many chapters.

The shared public API is defined by `projects/ch36/src/lib.rs:376-417,572`. Known callers include:

- `projects/ch38/src/lib.rs:68-88,525-526`
- `projects/ch39/src/main.rs:259-268`
- `projects/ch40/src/main.rs` through its Chapter 36 decoder integration
- `projects/ch41/src/main.rs:221-245`
- `projects/ch43/src/main.rs:26,40,111,170,324`
- Chapter 38, 39, 40, and 43 starters and associated tests

Independent implementations also use the shorter name in `projects/ch55/src/main.rs:33,114,149` and `projects/ch56/src/lib.rs:513,972,1351`. Training-rate parameters named `rate` occur in `ch31`, `ch33`, `ch43`, `ch44`, `ch46`, `ch48`, `ch50`, `ch51`, `ch52`, `ch55`, and `ch56`.

Root should either rename the full caller graph to the shared contract or revise the contract deliberately. The chapters should retain separate APIs where the algorithms differ; a universal ML trait would create superficial uniformity.

### High: Chapter 36 parameter spans are a de facto cross-crate ABI

`projects/ch36/src/lib.rs:329-357` publishes exact span names and a documented row-major `[input, output]` matrix layout. Chapter 40 looks up names dynamically (`projects/ch40/src/main.rs:21-122`); Chapters 41 and 43 hard-code `output_weight` (`projects/ch41/src/main.rs:221-245`, `projects/ch43/src/main.rs:15-21,111-129`). Any span rename, ordering change, or layout change must update every caller and starter together. `parameters_mut` is an intentional expert escape hatch, so callers must preserve span shape, orientation, and finiteness.

### Medium: Chapter 30 lesson has malformed HTML

`chapters/30/lesson.html:33` contains `</code` without the closing `>`. The remainder of the paragraph can be parsed as code markup in a browser.

### Medium: Chapter 33 overstates what Chapter 37 enforces

`chapters/33/lesson.html:7` says Chapter 37 “prevents” cross-document transitions. Chapter 37 parses, cleans, deduplicates, and assigns document splits (`projects/ch37/src/main.rs:28-65,85-105`), but it does not construct token windows or enforce boundary-safe traversal. The text should describe a document-boundary convention that downstream window builders must honor.

### Medium: Chapter 54 parser is not wire-compatible with Chapter 53

`chapters/54/lesson.html:30` says its parser “could be called by the preceding chapter’s server.” Ch53 accepts `schema=1&x=<scalar>` (`projects/ch53/src/main.rs:226-251`), while Ch54 accepts `schema=1,f0=<number>,f1=<number>` (`projects/ch54/src/main.rs:285-307`). This is a similar versioned-boundary pattern, but it requires an adapter and a different model input contract. The lesson should say so explicitly.

### Minor: private implementation named as if public in Chapter 38

`chapters/38/lesson.html:52` tells learners to compare against `Trainer::accumulate`, but that method is private (`projects/ch38/src/lib.rs:90-105`). The public entry point is `train_accumulated` (`projects/ch38/src/lib.rs:68-77`). Mark the former as an implementation detail or reference the public method.

## Chapter map and API transitions

- **29–32:** standalone GPU/device and resident-training examples; no shared ML model API.
- **33:** private bigram model with `loss`, `step`, and `predict`; byte-level next-token targets.
- **34–35:** standalone tokenizer and causal-attention/backward examples.
- **36:** reusable public `Decoder`; `forward`, `loss`, `loss_and_grad`, named parameter spans, and optional GPU variants. Uses explicit shifted targets supplied by callers and `[input, output]` matrix storage.
- **37:** standalone corpus parsing, cleaning, deduplication, and document splits.
- **38:** public `Trainer` over Ch36 with accumulated updates, AdamW, checkpoint/resume, and optional GPU route.
- **39:** public integration of Ch36 and Ch38; shifted byte targets, token-weighted held-out evaluation, generation, and resume checks.
- **40:** cached inference over Ch36; local cache `step` is token advancement, not an optimizer step.
- **41:** quantized output projection evaluated through a Ch36 clone; depends on `output_weight` span and `[D,V]` orientation.
- **42:** standalone tiled/online attention and benchmark example.
- **43:** Ch36 adaptation examples: full SFT, LoRA, and distillation; reuses Ch36 `loss_and_grad` and `output_weight`.
- **44–46:** standalone retrieval/reranking, RL, and DPO examples; their `train`/`step` methods have task-specific meanings.
- **47–52:** standalone quantization, MoE, recurrence, parallel training, generative models, and contrastive retrieval.
- **53:** standalone operational linear model with a scalar HTTP prediction contract.
- **54:** standalone two-feature audit/classifier with a different local parser contract.
- **55:** isolated scratch affine model plus framework parity; its `forward`, `loss_and_grad`, and `step` are not the Ch36 API.
- **56:** independent sparse decoder/MoE model with `forward`, `loss`, `loss_and_grad`, and `Trainer::train_step`; explicitly does not import Ch36 or Ch48.

## Reduction, shape, and target/label checks

No loss-reduction defect was found in the inspected paths. Ch36 averages cross-entropy over target rows; Ch38 documents equal-microbatch averaging; Ch39 token-weights unequal evaluation blocks; Ch48 and Ch56 distinguish training objectives containing auxiliary terms from uncapped task-only evaluation.

Shape conventions are explicit where they intentionally differ. Ch36, Ch40, Ch43, and Ch56 use `[input, output]` storage and explain it; Ch55’s scratch model uses `[output, input]` and explicitly transposes for its framework backend (`chapters/55/lesson.html:13-22`). Do not unify these layouts without preserving the stated mapping and checkpoint behavior.

`target(s)` is used for supervised next-token and soft targets, `label(s)` for discrete classification, and `reward` for RL. These transitions are semantically justified and should remain explicit rather than being flattened into one term.

## Validation note

The Chapter 53 serving command was checked against its parser and is correctly ordered. The report contains findings only; no re-audit or code changes were performed after the initial read-only audit.
