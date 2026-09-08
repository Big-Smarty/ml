# Section 07 review — Train your language model

Date: 2026-09-08. Reviewer: GPT-6 Astra Medium. Base: `43a16cc2c0b09a2bc4c75f79869cc2b9ab4fca5c`. Status: complete and frozen.

## Coverage

Read all seven complete lessons, metadata and research records under `chapters/33` through `chapters/39`; all reference and starter Rust sources and Cargo manifests under `projects/ch33` through `projects/ch39`; all fourteen matching Rustlings exercise/solution sources and their `info.toml` mappings. Also read Chapter 35's browser demo, Chapter 36's GPU integration test, and Chapter 39's bundled training/validation text. Consulted the consistency, numerics and authoring contracts, both applicable Rust skills, and existing review evidence without altering it.

| Chapter | Reviewed content and continuity |
| --- | --- |
| 33 | Byte versus Unicode n-grams, lookup/output shapes, stable mean cross-entropy, simultaneous embedding/output gradients, count-based starter. |
| 34 | Deterministic BPE tie order, non-overlapping replacement, ranked encoding, byte-exact decoding, serialization bounds and multilingual examples. |
| 35 | Causal row normalization, single/multi-head shapes, complete Q/K/V derivatives, upstream-gradient reduction, numerical worked backward, interactive mask. |
| 36 | Full pre-normalized decoder forward/backward, residual and repeated-embedding accumulation, GELU/LayerNorm derivatives, exact counts, stable loss, parameter spans, CPU/GPU bridge. |
| 37 | Parsing/CRLF, whitespace cleaning, identity, ordered shingle deduplication, whole-document split boundaries, audit provenance and detector limitations. |
| 38 | Token-weighted equal-microbatch accumulation, global clipping, AdamW moments/decay, schedule endpoints, transactional updates, complete checkpoint restoration and caller-owned resume choices. |
| 39 | Byte input, shifted targets, held-out tail weighting/history resets, generation, checkpoint CLI, explicit large-run budgets and real tiny training. |

Checked Chapter 30's matrix interface and Chapter 40's cached-decoder implementation/lesson against the unchanged Chapter 36 layout, including later Chapter 41/43 parameter-span consumers. The standalone tokenizer and corpus-audit boundaries remain explicit: the byte capstone does not claim to import either artifact automatically.

## Material findings and fixes

Two related Chapter 39 resume issues were fixed locally in `projects/ch39/src/main.rs` and documented in `chapters/39/lesson.html`:

1. A valid Chapter 38 checkpoint could contain a vocabulary other than 256, while the capstone interprets generated IDs with `as u8`. IDs above 255 would silently wrap instead of undergoing their tokenizer's decoding. Resume now requires vocabulary 256; custom byte-model widths and contexts remain accepted.
2. A large checkpoint resumed without `--large` bypassed the explicit step-budget guard and changed microbatches from eight targets to the saved 128-token context, while also removing the sixteen-target evaluation budget. Resume now rejects this combination and directs the caller to `--large --resume PATH --steps N`. The existing large-dimension check remains in force.

Before the guards were added, the new regression failed against the existing resume validation (exit 101). Its final form checks smaller/larger incompatible vocabularies, a large checkpoint without large mode, valid large mode, a tiny checkpoint incorrectly marked large, and valid custom byte-model dimensions. Large cases inspect only `Config`; they do not allocate the 14.4M model.

No other material correctness or teaching issue was found. No shared API, parameter layout, optimizer/checkpoint format, shader, starter, exercise, data artifact, or prior validation record changed.

## Checks and results

For each `NN` in `33 34 35 36 37 38 39`, ran:

```bash
cargo fmt --manifest-path projects/chNN/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path projects/chNN/Cargo.toml -- -D warnings
cargo test --offline --manifest-path projects/chNN/Cargo.toml
cargo run --release --offline --manifest-path projects/chNN/Cargo.toml
cargo fmt --manifest-path projects/chNN/starter/Cargo.toml -- --check
cargo clippy --offline --all-targets --manifest-path projects/chNN/starter/Cargo.toml -- -D warnings
cargo run --offline --manifest-path projects/chNN/starter/Cargo.toml
cargo test --offline --manifest-path projects/chNN/starter/Cargo.toml
```

All formatting, strict Clippy, reference tests and default runs passed. Final reference test counts: 33: 4; 34: 4; 35: 2; 36: 10; 37: 4; 38: 6; 39: 4. Each starter launched usefully and its one guided test failed only at its intended TODO (exit 101). Matching exercises and solutions compiled with `rustc --edition=2024 --test`; all seven exercises failed at their TODOs and all seven solutions passed.

Chapter 39's formatting, strict Clippy and tests were rerun after the change. Optional integration compilation also passed:

```bash
cargo clippy --offline --all-targets --features gpu --manifest-path projects/ch39/Cargo.toml -- -D warnings
```

A scoped HTML parser checked all 22 `data-source` excerpts in these seven lessons against their defining files, ignoring whitespace: all matched.

Bounded demo observations: Chapter 33 loss 5.545 → 0.957; Chapter 34 36 bytes → 12 tokens, vocabulary 268 and both round-trip checks true; Chapter 35 matches its worked probabilities/output; Chapter 36 loss 2.0803 → 0.0383; Chapter 37 retains three records and audits one removal; Chapter 38 restores step 26; Chapter 39 default processes 960 targets and reports held-out loss 5.545 → 4.472. These are fixture results, not language-quality or benchmark claims.

Using the built Chapter 39 release binary and a fresh temporary checkpoint, also ran `--steps 1 --generate 0 --checkpoint PATH`, then `--resume PATH --steps 1 --generate 0`: both succeeded and reported steps 0 → 1 → 2. `--large --resume PATH --steps 1 --generate 0` rejected the tiny checkpoint; `--large --resume PATH` rejected the absent explicit budget. `--large-info` reported exactly 14,442,496 parameters without allocation.

Transient detailed logs are `/tmp/section07-{fmt,clippy,test}-NN.log`, `/tmp/section07-NN-extra-{0,1,2,3,4}.log`, and `/tmp/section07-NN-{exercises,solutions}.log`.

## Limits and handoff

No GPU hardware execution, large-model allocation/training, extended training, global site build/verifier, or Git mutation was performed. Existing hardware evidence is preserved; the changes affect only CPU-side CLI validation. Root owns integrated publication checks. No outstanding cross-section interface issue remains, and no shared signature changed.
