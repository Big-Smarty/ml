# Train a tiny byte language model

This independent Cargo package carries chapters 33–39. Every default command works offline on CPU. The baseline is a useful simpler algorithm; passing it does not complete the learning goal. `--check` invokes your functions and initially exits 1 with `GOAL_NOT_MET:` and numerical evidence. Input/runtime failures have ordinary error messages. Separate solutions do not overwrite learner files.

```bash
cargo run --offline --manifest-path labs/s07-language-models/Cargo.toml -- 33
cargo run --offline --manifest-path labs/s07-language-models/Cargo.toml -- 33 --check
cargo run --offline --manifest-path labs/s07-language-models/Cargo.toml -- 33 --solution --check
```

From the repository root, `just lab 33`, `just lab-check 33`, and `just lab 33 --solution --check` are equivalent entry points. Replace 33 with 34–39. CLI experiment options follow the chapter number. Chapters 33, 35, 36 and 38 accept only the standard flags; their documented variations edit the named fixture/configuration values below.

| Chapter | Working baseline | Your core implementation | Evidence and transfer |
|---|---|---|---|
| 33 | Uniform count predictor plus trained output biases | `count_loss`: smoothed conditional counts; `update`: full embedding/output/bias derivatives and simultaneous update | New successor has probability 1/258; f64 parameter probes; fit previously unseen alternating byte IDs |
| 34 | Exact raw-byte tokenizer and serialization | Full document-aware BPE `fit`, ranked `encode`, bounded exact-byte `decode` | Ties, overlapping pairs, separated documents, multilingual and arbitrary-byte round trips, reload ID equality |
| 35 | Causal uniform value averaging with its corresponding value derivative | Scaled QK softmax, weighted values and full Q/K/V backward | All elements of an unfamiliar [3,2] fixture pass central differences; future changes cannot affect earlier rows |
| 36 | Embedding/output learner forward with identity blocks; supplied full decoder actually trains | Complete pre-normalized attention/FFN `block`, plus LayerNorm input/gain/bias backward | Two-layer forward parity, causal prefix, normalization derivatives; supplied full-backward reference remains available |
| 37 | Fixed cleaning, exact dedup, stable document splits | Extend `prepare` with unique word-trigram Jaccard and auditable near-duplicate decisions | Unfamiliar exact/near fixture, source/license/removed links; compare thresholds on reviewed documents |
| 38 | Actual decoder training with one microbatch mean and fixed-rate SGD | Token-weighted `combine`, warmup/cosine `rate`, globally clipped AdamW `update` | Unequal tails, successive moments, transaction validation, exact next-step resume and changed-data rejection |
| 39 | Real full decoder training on first document/one microbatch; explicitly labelled first-block evaluation | Integrate all documents/two microbatches and completed ch38 algorithms; weight all evaluation targets | All-parameter training, nontrivial loss decrease, two-document integration, generation and exact selected-callback resume |

The cumulative trainer calls learner chapter 38 update/combination functions through chapter 39. Complete them before integrating the chapter 39 learner. The solution route uses only the separate completed functions. The full decoder backward is supplied by the preserved `projects/ch36` library; chapter 36 opens forward composition and normalization backward instead of asking you to reconstruct storage or every derivative at once. Its full architecture and backward are explained in the lesson and remain tested in the reference.

## Experiment controls and checkpoints

Save the standard output, exact source/configuration and data for each checkpoint. Keep a named copy of a completed learner file before the next experiment (for example `cp labs/s07-language-models/src/ch38.rs /tmp/ch38-adamw.rs`). A copy is a local convenience, not a durable experiment archive. Use your normal source-control workflow for durable history; no script silently edits or restores your files.

- **33:** `byte_model.rs::TEXT`, `run_with`'s `valid`, `Model::new(4)`, 80 updates and rate 0.8 are the actual inputs. Change width to 2/8 while holding text/seed/update count fixed. The seed is 33. Report the actual count and neural validation losses.
- **34:** `--merges 0|8|24` changes the actual requested fit budget. `tokenizer.rs::run_with` owns the named `train` and `held` strings. The raw-byte baseline deliberately learns no rules. The completed default learns 12 rules, compresses 36 training bytes to 12 tokens and 17 held-out bytes to 16 tokens.
- **35:** `attention.rs::fixture()` owns Q/K/V/T/D. Change one future key or value; keep prefix outputs fixed. Use two independent width-two `Inputs` fixtures for the multi-head transfer exercise.
- **36:** `decoder.rs::config()` owns the two-layer model configuration; `run_with` owns input/target, 40 SGD updates and rate 0.08. Preserve width/head divisibility when changing depth. Restore standard values before recording the standard check.
- **37:** `--corpus documents.txt --threshold 0.6` (also 0.8/0.95). Every document has `Title: ...`, `License: ...`, then text; separate records by a line `===DOC===`. Source path and license remain attached. At most 64 KiB. The bundled hash split produces three train documents and zero validation/test: that is not an evaluation dataset.
- **38:** `training.rs::tiny_trainer()` owns seed, architecture and `TrainConfig`; `run_with` owns `docs` and 24 updates. Defaults: peak 0.02, minimum 0.002, two warmup updates, forty-update horizon. Alter only one value and retain both records.
- **39:** flags below change the actual run. The horizon is stored in the trainer configuration; `--steps` is additional updates, not a request to restart the schedule. The fixed sample policy lives in `capstone.rs::run_with`: prompt `The `, 32 new bytes, independent seed 390 and temperature 0.8.

```bash
just lab 34 --solution --merges 8
just lab 37 --solution --corpus documents.txt --threshold 0.8
just lab 39 --solution --context 4 --steps 40
just lab 39 --solution --context 8 --steps 20
just lab 39 --solution --steps 20 --checkpoint run.bin
just lab 39 --solution --resume run.bin --steps 20 --checkpoint run.bin
just lab 39 --solution --corpus train.txt --validation valid.txt --steps 40
just lab 39 --large-info
```

For timing, use `cargo run --release --offline --manifest-path labs/s07-language-models/Cargo.toml -- 39 --solution ...`. The printed single-run throughput includes gradient/optimizer allocation and excludes evaluation and checkpoint I/O. It is not a stable performance benchmark. Compare actual target counts: shorter document tails can make `steps × context × microbatches` inaccurate.

## Data and artifact contract

`data/train.txt` and `data/validation.txt` are the same course-authored CC0 documents preserved in the chapter 39 reference (613 and 689 bytes on delivery). They stay separate. Custom CLI files are one raw-byte training document and one raw-byte validation document, each at most 64 KiB. Exact equality and shared 32-byte passages are rejected; this does not detect paraphrases. The in-memory training APIs accept multiple separate documents. The chapter 37 audit is retained separately, not parsed by the capstone.

The capstone is byte-level (vocabulary 256). Chapter 34 is a full BPE comparison lab; no BPE integration into the model is required. Parameter shapes and `CH38LM02` serialization are reused unchanged. The new window cursor and fingerprint include document boundaries, so exact training continuation must use this lab's traversal and microbatch policy. Inference consumers can use the compatible decoder parameter layout. Checkpoints exclude source code, corpus files, manifests and call-level choices; retain them alongside the checkpoint. File saves use the existing checked temporary-write/rename path. Same-executable next-update equality is tested; cross-platform floating-point bit identity is not promised.

Defaults use a 4,932-parameter CPU decoder with context 4. Context 8 has 4,964 parameters. At most 200 requested updates run without `--extended`; that flag raises the hard teaching ceiling to 2,000. Tiny resume files are bounded to 250,000 bytes before loading. `--large-info` prints the real 14,442,496-parameter configuration without allocating it. Full large/GPU execution stays in the explicit preserved reference commands in the chapter; `--gpu` here returns an unsupported result. Nothing downloads data or starts GPU work implicitly.

## Verification

```bash
cargo fmt --check --manifest-path labs/s07-language-models/Cargo.toml
cargo clippy --offline --manifest-path labs/s07-language-models/Cargo.toml --all-targets -- -D warnings
cargo test --offline --manifest-path labs/s07-language-models/Cargo.toml
node labs/s07-language-models/check_interactives.cjs
```

The supplied test suite passes with the intact baselines and separately tests solution numerics. Goal checks fail until the corresponding learner code is implemented. Numerical evidence cannot assess the written comparison, provenance reasoning or explanation of a failure; each session's transfer prompt remains a self-assessed part of the work.
