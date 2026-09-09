# Section 09: advanced models and systems

Run a complete small baseline, implement the chapter's substantial algorithms, and compare actual numerical behavior:

```sh
just lab 47
just lab-check 47
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 47 --solution --check
```

Replace 47 with any chapter 47–56. All defaults are deterministic, CPU-only, standard-library Rust experiments with no network, socket, dataset download, or optional framework dependency. This package is an independent Cargo workspace.

`src/chNN.rs` is the learner entry. Its baseline works but does not satisfy the complete learning goal. `src/solutions/chNN.rs` is the separately commented solution. `src/common/chNN.rs` supplies fixture definitions, validation, or incidental I/O; `src/checks/chNN.rs` runs goal probes against the actual selected implementation. The lessons explain intermediate values, implementation boundaries, graduated hints, and changed-input transfer exercises. The preserved `projects/chNN` programs are reference implementations and are not modified by these labs.

| Chapter | Working baseline | Whole learner algorithm and evidence |
|---|---|---|
| 47 | Retain weights and store explicit zeros | Global deterministic magnitude mask, zero-omitting CSR, indirect sparse matvec; separate pruning error, parity, bytes, and repeated kernel time |
| 48 | Fixed router with trainable experts, no capacity | Complete joint router/expert update, full-softmax gate gradient, capacity admission, attempted-frequency balancing |
| 49 | Serial affine prefixes and uniform causal average | Ordered offset-doubling scan and normalized feature-kernel summaries; direct oracle, future-value probe, source review of scan schedule |
| 50 | Worker-weighted means and frozen first pipeline stage | Example-weighted sum/count reduction and full backward/update through real thread/channel tensor and pipeline paths; serial and recovery comparisons |
| 51 | Deterministic autoencoder, minimax GAN, identity reverse sampler | Reparameterized beta-VAE objective, stable non-saturating GAN, noise-prediction objective and full deterministic reverse chain |
| 52 | Paired embedding attraction without negatives | Normalized dual-encoder pair scores and symmetric row/column cross-entropy; both towers update and retrieval uses explicit candidates |
| 53 | Reject every candidate; retain cumulative history | All-canary release decision and bounded four-request recent-input monitor; validated preprocessing/artifact/request path |
| 54 | Pooled accuracy and selection-rate report | Group-filtered confusion accounting and undefined denominators; explain supplied ablation, endpoint perturbation, membership, parser, and association controls |
| 55 | Reshaped inference-only runtime; aggregate one-input comparison | Layout adapter, independent forward and full gradient/update implementation, coordinate diagnostics and complete parity workflow; actual file export/import |
| 56 | Contextual decoder always using expert zero | Full sparse expert forward/backward inside the complete decoder, actual training, dense comparison, uncapped causal evaluation, recovery, bounded serving |

Intentional learning-goal comparisons return `GOAL_NOT_MET:` and exit 1. Input, filesystem, parser, and runtime errors retain ordinary error messages. Passing numerical checks for learner chapter 49 returns `GOAL_REVIEW_REQUIRED:` and exit 3, because source review must verify offset-doubling rounds with immutable previous-round reads and ordered composition. Its metadata records the review rubric; the author-reviewed solution returns zero. A passing executable does not assess the explanation required by each lesson's transfer task.

Every extra argument is part of an explicit command, not an ignored experimental flag. Chapters 47–52 and 54 take no extra experiment arguments. Independent variations name exact editable source values or ask for a new learner-owned probe.

## Optional artifact and serving commands

Chapter 53 trains an artifact or starts an interactive loopback server:

```sh
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 53 train /tmp/s09-model.txt
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 53 serve 127.0.0.1:8787 /tmp/s09-model.txt
```

Stop that interactive server when finished. Its default offline experiment opens no socket. The existing loopback integration test is opt-in:

```sh
cargo test --manifest-path labs/s09-advanced/Cargo.toml serving_monitor_and_rollback_are_real -- --ignored
```

Chapter 55 exports only to a new path and validates imported bytes. Its separate optional Burn path may download pinned dependencies; it runs the preserved reference's own 2×2 fixture, not the active lab's 72-byte format.

```sh
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 55 export /tmp/s09-affine.bin
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 55 import /tmp/s09-affine.bin
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 55 --framework
```

After completing chapter 56, use a fresh artifact path:

```sh
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 56 train /tmp/s09-capstone.bin 160
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 56 eval /tmp/s09-capstone.bin
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 56 serve-check /tmp/s09-capstone.bin
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 56 resume /tmp/s09-capstone.bin 20
cargo run --manifest-path labs/s09-advanced/Cargo.toml -- 56 generate /tmp/s09-capstone.bin rust 8
```

Add `--solution` to run reference algorithms. `serve-check` loads the checkpoint, opens an ephemeral loopback port, sends one real request, checks a 200 generation response, and exits with bounded waits. `serve CHECKPOINT [ADDRESS]` accepts one interactive request or times out waiting after 30 seconds. Custom training requires both explicit training and held-out ASCII files: `56 train CHECKPOINT STEPS TRAIN_TEXT HELDOUT_TEXT`. Extended steps are explicit and bounded to 1–100,000 per invocation. No public deployment is performed.

## Author verification

```sh
cargo fmt --manifest-path labs/s09-advanced/Cargo.toml --check
cargo clippy --manifest-path labs/s09-advanced/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s09-advanced/Cargo.toml
node chapters/48/demo-check.cjs
```

The package tests pass on supplied baselines and plumbing while solution tests assess completed algorithms. They do not pretend that the unfinished learner goals pass. The section design matrix and actual observed numerical evidence are in `guidance/redesign/section-09.md`.
