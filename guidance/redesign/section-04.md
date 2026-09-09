# Section 04 — broader deep learning redesign

Owner: one GPT-6 Astra High section author. Completed on 2026-09-09. Scope: chapters 20–24, `labs/s04-deep-learning`, this report. No chapter implementation was delegated; no reference project, shared site, catalog, publication or git history was changed by this owner.

The owner read the active AUTHORING/CHAPTER_TEMPLATE/ASSIGNMENTS/NUMERICS contract, complete assigned old lessons/metadata/research, all five reference implementations and their starters, the chapter 01 pilot interface, both Rust skills, and the supplied systems/interactive/learning-science evidence syntheses. Existing primary-source verification settled the technical claims; no additional research delegation or new experimental claim was needed.

## Exact original-topic coverage and implementation matrix

| Chapter | Original topics preserved verbatim → teaching steps | Observable learner objective | Actual algorithmic change | Check and fresh transfer |
|---|---|---|---|---|
|20|Convolution→`local-weights`; padding and stride→`shape`; pooling→`pooling`; gradients→`gradients`|Build/train a small convolutional digit classifier and explain shared derivatives|Replace pointwise convolution with a whole padded shared map, replace upper-left subsampling with max values/winners, accumulate all nine spatial weight gradients|4×5 asymmetric-kernel outputs, unique 2×2 winner, nine central differences, trained digit loss/accuracy; transfer to odd 3×5 pooling, a corner derivative and a new 6×8 stroke image|
|21|Residual connections→`residual`; normalization→`normalization`; augmentation→`augmentation`|Differentiate normalized residual branches and compare actual training translations|Replace centering with per-image mean/variance scaling, implement coupled input backward, implement signed zero-padded translation|All36 normalized values, three input derivatives, first kernel through two residual blocks, impulse(1,1)→(2,0); transfer to constant/shifted arrays and cropped corner impulses|
|22|Autoencoders→`autoencoder`; bottlenecks→`meaning`; contrastive learning→`contrastive`|Train an encoder through its bottleneck, differentiate both shared paired views, choose cosine neighborhoods|Replace frozen-encoder gradients, extend anchor-only contrastive derivatives to candidates, replace raw Euclidean selection with epsilon-aware cosine|Three AE derivative probes on an unfamiliar vector, asymmetric three-pair batch at τ=0.37, both objectives decrease, unequal-length and zero-vector neighbor cases; compare candidate geometry and task objectives|
|23|RNNs→`state`; BPTT→`bptt`; LSTMs→`lstm`; temporal evaluation→`evaluation`|Unroll state, compute full RNN/LSTM BPTT and implement a forecast consuming its own predictions|Extend a one-lag predictor to recurrence and all shared gradients; replace length-one LSTM truncation with hidden/cell derivative propagation; replace persistence deployment with generated-prefix rollout|Signed-sequence state trace, RNN input/recurrent probes, LSTM candidate/forget probes, seed 0.63 two-step feedback/state check, later-target loss; transfer to reversed inputs, zero recurrent weight and horizon 1/10/20 reports|
|24|Embeddings→`embeddings`; matrix factorization→`factorization`; ranking metrics→`metrics`|Learn both factor tables, evaluate multi-relevant rankings and complete a controlled section experiment|Extend item-only training to simultaneous user/positive/negative updates; replace un-discounted gain with general binary nDCG|Six coordinate-update derivatives on a new triple/rate/regularizer, multi-relevant top-k denominators/discounts, holdout exclusion and improved ranking; transfer to changed k, swapped preferences and rotated explicit negatives|

Substantive original content also remains: image geometry/channels/receptive fields, ReLU/max tie conventions, dense softmax/cross-entropy and complete CNN training, optional local MNIST IDX; identity/projection distinction, full two-block residual backward, per-image versus batch normalization, label-preserving augmentation and source-level leakage; reconstruction versus denoising targets, bottleneck ambiguity, stable InfoNCE/temperature, normalization derivative and retrieval; modern four-gate LSTM and historical forget-gate distinction, vanishing/exploding gradients, clipping/truncation, persistence and free-running versus teacher-forced forecasting; pairwise logistic/L2 objective, explicit versus unknown negatives, deterministic candidate order, Recall/Precision/HitRate/nDCG distinctions, cold start and bias cancellation.

All five lessons are new reader-oriented compositions. Meaningful original anchors remain as learning steps or nested headings (`continuity`, main topic anchors, `practice`, `review`, `sources`; chapter21's `build`, chapter22/24's `experiment`, and chapter 20's `run` remain). Source excerpts point to exact new solution paths with `data-source`. Glossary links retain immediate definitions and use canonical `.term`/`data-term` markup.

## Session map and exit evidence

Each group below is a coherent 30–45-minute session, with explanations, explicit numerical work, a named implementation or controlled variation, optional graduated hints/full answer links, and a new-case explanation. Navigation is never gated.

| Chapter/session | Ordered steps | Minutes | Local work and exit |
|---|---|---:|---|
|20/1|local-weights,shape,convolve|40|Whole convolution map; `--checkpoint convolution`; rectangular boundary and impulse trace|
|20/2|pooling,routing,reduction|35|Max values/indices; `--checkpoint pooling`; odd-dimension and moved-winner trace|
|20/3|gradients,backward,debug|38|Nine shared derivatives; `--checkpoint gradients` and full check; corner padding explanation|
|20/4|run,mnist,review|35|Replace max with top-right subsampling for one controlled run, restore/check; new 6×8 image and explicit optional MNIST shapes|
|21/1|residual,normalization,statistics|40|Variance scaling; `--checkpoint statistics`; constant and shifted input checks|
|21/2|backward,build,practice|38|Normalization backward; `--checkpoint backward`; constant incoming-gradient property and two-route diagnosis|
|21/3|augmentation,translations,review|40|Translation; `--checkpoint augmentation`; `--canonical` ablation with equal default update counts and corner/cropping transfer|
|22/1|autoencoder,meaning,reconstruct|40|Encoder gradient; `--checkpoint autoencoder`; zero features versus bias derivatives|
|22/2|contrastive,shared-gradient,train-contrastive|40|Both paired branches; `--checkpoint contrastive`; asymmetric/duplicated-batch reasoning|
|22/3|neighbors,experiment,review|37|Cosine selector; `--checkpoint neighbors`; unequal norms, zero vectors, new query and exact temperature-call variation|
|23/1|state,unroll,targets|38|Recurrent forward; `--checkpoint states`; reversed signed inputs and target/cache alignment|
|23/2|bptt,reverse,credit|40|Full RNN reverse; `--checkpoint bptt`; zero recurrent weight and deliberate local-only comparison|
|23/3|lstm,lstm-backward,memory|40|Both LSTM temporal derivatives; `--checkpoint lstm`; fixed-gate retention and named forget-bias ablation|
|23/4|evaluation,rollout,review|40|Generated-prefix rollout; `--checkpoint rollout`; real `--horizon` comparison with per-horizon errors|
|24/1|embeddings,factorization,pairwise|40|Three-vector update; `--checkpoint update`; swapped preference and collaborative exclusion reasoning|
|24/2|split,metrics,metric-code|40|Multi-relevant binary nDCG; `--checkpoint metrics`; k3→4 transfer|
|24/3|experiment,sampling,review|40|`--rotate-negatives` comparison, metric-code comparison and chosen section experiment with named controls|

Total: 17 sessions, 51 reader steps, 661 estimated minutes. These are planning estimates rather than enforced times. Early checkpoint commands remain available after later algorithm edits; partial forward changes are checked directly before training resumes with a matching backward rule.

## Package interface and supplied scope

The independent dependency-free Cargo package uses Rust 2021 and its own `[workspace]`. Every chapter has an actual learner module and a separate explained solution module. Small `Core` structs select numerical functions in the same supplied model/trainer/check path. This does not dispatch to an unrelated success demo when checking learner work.

- `cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- NN` runs the selected useful baseline.
- Add `--check` for full selected-core goals; numerical/algorithm mismatches are `GOAL_NOT_MET:` with exit 1.
- Add `--solution` for the complete selected core; solution full/stage checks exit 0.
- Add `--checkpoint NAME` for an early cumulative checkpoint; all 19 named solution checkpoints were run successfully.
- Unknown chapter/stage/argument, invalid horizon and missing IDX paths remain ordinary errors, not prefixed goal failures.
- `just lab NN` / `just lab-check NN` remain the public course shortcuts supplied by root.

Actual variation controls:20 `--mnist` with four local paths;21 `--canonical` prints actual image updates, with both default policies using 700;23 `--horizon 1..40` prints per-horizon target/RNN/LSTM-deployment/persistence squared errors;24 `--rotate-negatives` prints actual triples/settings/ranks. Other variations name an exact editable source function or literal: `ch20::pool`, `ch22::nearest`, all matching temperature calls in `representation::run`, forget bias in `sequence::Lstm::new`, and the shift list in `residual::data`. No promised control is silently ignored.

Complete underlying algorithms are adapted from the already-tested reference implementations. The new ownership boundary exposes mathematical cores while preserving real CNN training, two residual blocks, AE and shared InfoNCE training, full modern LSTM forward/backward and factorization. The original `projects/ch20`…`projects/ch24` are unchanged. Fixed numerical fixtures use assertions for internal invariants; the actual external boundary consists of validated CLI options and the defensive IDX reader.

## Measured local outcomes

All values below come from the completed package's tiny offline runs, using its stated fixed budgets.

| Chapter | Supplied learner baseline | Completed solution |
|---|---|---|
|20|Validation CE2.3026→1.8473; accuracy1/10→3/10|CE2.3062→0.0630; accuracy1/10→10/10|
|21|Centered/canonical repeated views: CE0.6949→0.0266;2/4→4/4|Standardized translated views: CE0.6949→0.0928;2/4→4/4; scales[0.5993,0.3146]|
|22|AE MSE0.38879→0.06391; InfoNCE1.22625→0.80896; paired retrieval1/6→2/6|AE MSE0.38879→0.00252; InfoNCE1.22625→0.79667; paired retrieval1/6→3/6; unfamiliar reconstruction MSE0.002226|
|23|One-lag later MSE0.41995→0.01618; truncated LSTM0.39095→0.06187|Full RNN0.39891→0.01194; LSTM0.39095→0.01779; persistence0.01390. RNN10-horizon mean squared error0.09635|
|24|Pairwise loss0.69065→0.62627; Recall@2 falls0.75→0.50|Loss0.69065→0.02021; Recall@2→1.0, Precision@2→0.5, nDCG@2→1.0|

These are not fabricated larger-data results. The residual baseline already performs strongly; the completed LSTM still loses to persistence on one-step validation; representation retrieval remains imperfect; the item-only recommender can lower fitting loss while worsening recall. The lessons explain those outcomes instead of equating more architecture, lower loss or successful execution with learning/usefulness. Optional MNIST was not downloaded or newly scored in delivery; synthetic IDX reader tests passed.

## Interactive tools and deterministic evidence

Exactly two chapter-local scripts are declared in metadata. Both scope DOM queries to one unique container, use native labelled selects/buttons, have deterministic Reset and live textual numerical outputs, retain static/no-script equivalents, make no external requests, and never execute Rust or record completion automatically. Neither uses color to convey necessary information or requires dragging/autoplay.

1.20 `convolution-microscope`: fixed 5×5 image with center column [0.2,0.4,0.6,0.8,1], fixed 3×3 edge kernel, bias 0. Default valid convolution outputs [1.2,0,−1.2;1.8,0,−1.8;2.4,0,−2.4]; ReLU clamps negatives. First 2×2 max winner is 1.8 at (1,0). Incoming pool derivative 0.6 gives kernel contributions [0,0,0.24;0,0,0.36;0,0,0.48]. Padding 1 gives 5×5; padding 1/stride 2 gives 3×3 and explicitly blocks the winning zero through ReLU. Step buttons expose every selected patch's nine products.
2.22 `embedding-neighborhood`: query [1,0], items A[1,0],B[0.8,0.6],C[0,1],D[−1,0]. Excluding observed A gives order B,C,D, Recall@2=1,Precision@2=0.5,nDCG@2=1. Query[0,1] gives C,B,D and nDCG 0.6309. Scaling only C to [0,3] changes its vertical-query dot 1→3 and distance 0→2 while cosine stays 1. A fixed decoder maps query [0,1] to [0,1,1,−1]. The geometry is explicitly designed, not a plotted training result.

Runnable check: `node labs/s04-deep-learning/check-demos.cjs`. It executes the actual browser scripts with a minimal DOM and asserts geometry, products, ReLU, pool/backward route, dot/cosine/distance, candidate exclusion, metrics, fixed reconstruction and reset states. This verifies deterministic calculations/event wiring, not rendered mobile accessibility; site-level browser/layout inspection remains part of root integration.

## Validation performed

- `cargo fmt --manifest-path labs/s04-deep-learning/Cargo.toml --check` — passed.
- `cargo clippy --manifest-path labs/s04-deep-learning/Cargo.toml --all-targets -- -D warnings` — passed without lint suppressions.
- `cargo test --manifest-path labs/s04-deep-learning/Cargo.toml` —31 passed, zero failed. Tests retain solution finite differences, complete training, stable extreme logits/softplus, IDX and finite-row boundaries, temporal exclusion, all baseline runs, all full solution goals, unknown options and horizon boundaries.
- 49 explicit CLI executions passed their expected outcomes: five normal learner baselines; five learner full goals with exit 1 and `GOAL_NOT_MET:`; five complete solution runs; five full solution checks ; 19 stage solution checks; five bad-stage ordinary errors; four concrete variation runs; one missing-IDX ordinary error.
- `node labs/s04-deep-learning/check-demos.cjs` — passed.
- Focused content audit verifies ordered step/session mappings, exact original topics, all source excerpts, source/solution/report presence and interactive metadata. Result: zero findings for each of20–24;51 ordered steps,17 sessions,17 exact catalog topics and two declared interactives. Word counts were3203,2446,2856,2984,2661 respectively; no minimum-word padding was used.

The unfinished learner failures were numerical evidence:20 missing spatial coefficient contributions;21 inverse scale 1 versus 1.920062507;22 frozen encoder derivative 0 versus −0.035455883;23 second state −0.119427299 versus −0.107596259;24 frozen user derivative 0 versus 0.039153433. None is an intentional startup panic or `todo!`.

## Prerequisite audit and section integration

The catalog order and prerequisites are unchanged:20 follows 19, then 21→22→23→24. Every newly used ML idea is introduced at its point of use rather than assumed from general Rust familiarity.

- **20** briefly retrieves dense layer shapes, stable classification loss and shared-value chain rule from earlier neural chapters before introducing locality, cross-correlation, receptive fields and spatial derivative accumulation.
- **21** retrieves additive/relu derivative routing from 20; defines per-image statistics and both residual routes before the two-block check. It distinguishes architectural residuals from regression errors and does not smuggle in batch-normalization running state.
- **22** retrieves dense shape/old-weight backward conventions; defines encoder/decoder targets, coordinate MSE, bottlenecks, positive/negative pairing, temperature, both shared encoder paths and norm derivatives. Recommendation metric previews receive concrete static values and are derived fully in 24.
- **23** retrieves shared-parameter accumulation from 22 and split discipline from earlier evaluation chapters. It defines state/cache/target indices before BPTT, modern LSTM gates before their two derivative paths, and observed-input/free-running protocols before comparing them.
- **24** retrieves embedding geometry from 22 and target exclusion from 23. It introduces lookup IDs, score-matrix shapes, pairwise derivatives, multiple-relevance denominators and candidate restrictions before asking for a controlled section project.

The section project chooses one of the actually implemented image/representation/forecast/recommendation comparisons and records input/control, settings, metric denominator, a fresh case and a causal explanation. It does not force disparate domains into one artificial dataset. Numerical equality does not claim source-level mastery; the lessons retain self-explanation/transfer requirements without inventing an automatic mastery gate. No structural-only goal requires an exit 3 review path here.
