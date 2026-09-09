# Broader deep learning: representations across data types

Five offline experiments share one CLI and reporting pattern. You implement each architecture's mathematics in `src/ch20.rs` through `src/ch24.rs`; the supplied experiment modules handle fixtures, model storage, updates and reporting. Complete, explained numerical implementations live separately in `src/solutions/`.

From the repository root:

```sh
just lab 20
just lab-check 20
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 20 --solution
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 20 --solution --check
```

Replace 20 with any chapter 20–24. A normal run is useful before editing. `--check` invokes the selected learner functions; expected unfinished-goal comparisons print `GOAL_NOT_MET:` and exit 1. Input/runtime failures remain ordinary errors. A solution check verifies the solution file, not your edits. Passing supplied tests or moving a browser control is not a claim of mastery.

## Baselines and mathematical work

| Chapter | Supplied working baseline | Learner change | Supplied experiment |
|---|---|---|---|
|20|Pointwise detector and upper-left subsampling|Whole shared padded convolution, max values/winners, accumulated kernel derivatives|Two-filter CNN, ReLU, ten-logit head, SGD, generated digits, optional IDX reader|
|21|Centered residual branches, repeated canonical views|Per-image standardization, coupled backward, zero-padded translation|Two complete residual convolution blocks and an equal-update augmentation comparison|
|22|Fixed encoder with trained decoder; anchor-only contrastive update; Euclidean neighbors|Encoder gradient, both shared InfoNCE branches, epsilon-aware cosine selection|4→2→4 tanh autoencoder and shared4→2 normalized contrastive encoder|
|23|One-lag nonlinear predictor; one-step truncated LSTM gradients; persistence deployment|RNN unroll/full BPTT, LSTM hidden/cell credit, generated-prefix rollout|Complete gated LSTM forward, chronological split, updates, one-step and per-horizon reports|
|24|Item-only factor update and un-discounted normalized gain|Simultaneous user/positive/negative update and binary nDCG|Two-factor recommendation, fixed explicit negatives, candidate exclusion and ranked-ID reports|

Baselines have complete derivatives for their declared model or an explicitly declared stop-gradient/truncated update. When changing forward before backward, use the small checkpoint rather than interpreting a mixed-model training run. The zero AE encoder gradient is an intentional fixed random feature model: its decoder still learns.

## Session checkpoints

```sh
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 20 --checkpoint convolution
```

All checkpoints are cumulative within their chapter and call the selected core. Add `--solution` to inspect the completed reference. Earlier checkpoints remain callable after later edits.

| Chapter | Checkpoint names in order |
|---|---|
|20|`convolution`, `pooling`, `gradients`, `training`|
|21|`statistics`, `backward`, `augmentation`|
|22|`autoencoder`, `contrastive`, `neighbors`, `training`|
|23|`states`, `bptt`, `lstm`, `rollout`, `training`|
|24|`update`, `metrics`, `training`|

Full `--check` also includes the later training goal when specified. The check functions are readable in `vision.rs`, `residual.rs`, `representation.rs`, `sequence.rs` and `recommendation.rs`. They compare arithmetic, routes and gradients using tiny unfamiliar inputs; they do not call a hidden learner-independent success demo. Numerical derivative tolerance is absolute 1e-6 plus relative 1e-4, with central-difference epsilon 1e-5 in f64. Max/ReLU gradient fixtures avoid ties and kinks; separate forward checks state tie conventions.

## Concrete experiment controls

```sh
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 21 --canonical
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 23 --horizon 20
cargo run --manifest-path labs/s04-deep-learning/Cargo.toml -- 24 --rotate-negatives
```

- **21** compares four canonical images with twenty requested shifted views. Both default policies use 700 image updates and print the actual update count. The baseline translation is identity until you implement it.
- **23** accepts horizon 1–40, reports squared error at each horizon, and keeps teacher-forced one-step evaluation separate. The completed LSTM rollout replays a generated prefix, O(horizon²), bounded to 40. A stateful step interface is the upgrade for long forecasts.
- **24** changes only which supplied explicit negative pairs with each positive; it prints actual triples, settings and candidate lists. It never samples the user's held-out positive.

The lessons also name exact editable functions/settings for independent changes: `ch20::pool`, `ch22::nearest`, the temperature calls in `representation::run`, the forget bias in `sequence::Lstm::new`, and the `shifts` list in `residual::data`. No undocumented argument is silently ignored.

The optional CNN extension uses prepared local MNIST IDX files:

```sh
cargo run --release --manifest-path labs/s04-deep-learning/Cargo.toml -- 20 --mnist \
  datasets/downloads/mnist/fit-images-idx3-ubyte \
  datasets/downloads/mnist/fit-labels-idx1-ubyte \
  datasets/downloads/mnist/validation-images-idx3-ubyte \
  datasets/downloads/mnist/validation-labels-idx1-ubyte
```

It validates headers, lengths, dimensions, overflow and labels; caps each split at 2,000 rows; trains one epoch at learning rate 0.01. No files are downloaded. Delivery tests use generated data and synthetic IDX payloads; no new MNIST benchmark is claimed.

## Data and interpretation

- **20:** thirty generated 8×8 seven-segment digits, ten dimmed/perturbed validation arrays. Related procedural sources limit generalization claims.
- **21:** two bar orientations and two thicknesses; five shifts request twenty training views; four validation perturbations use the same source patterns.
- **22:** six clean four-coordinate vectors and six paired views. Main reconstruction/retrieval reports are training-set evidence. The gradient and neighbor checks use different vectors and lengths.
- **23:**121 samples from a stated sine/cosine formula. Targets1–80 fit the model ; 81–120 are later. The boundary observation is context, not a future fitting target.
- **24:** four users, six items, eight fitting triples, one held-out positive per user, three explicit negatives per user. Ranking evaluates four candidates, not the full catalog. Binary multi-relevant metric checks extend beyond this special one-positive fixture.

All source datasets above are course-authored except optional MNIST. Complete original algorithms remain preserved under `projects/ch20`…`projects/ch24`; the new experiment modules adapt those algorithms with explicit learner-core selection. There are no third-party crate dependencies.

## Maintainer checks

```sh
cargo fmt --manifest-path labs/s04-deep-learning/Cargo.toml --check
cargo clippy --manifest-path labs/s04-deep-learning/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s04-deep-learning/Cargo.toml
node labs/s04-deep-learning/check-demos.cjs
```

The Node check executes the two real chapter scripts against a minimal DOM and asserts convolution/pool/backward values, embedding geometry/ranking metrics and reset behavior. It does not replace browser accessibility/layout inspection. Section coverage, session mapping, numerical results and prerequisite audit are recorded in `guidance/redesign/section-04.md`.
