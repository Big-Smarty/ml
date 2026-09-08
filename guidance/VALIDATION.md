# Final validation record

Course generation and subsequent checks on 2026-09-08. All 56 chapters are authored as HTML, with reference projects, runnable learner checkpoints, intentional exercises, complete solutions, worked examples, review questions, glossary explanations, and primary-source references. During the original course-generation pass, GPT-6 Astra High completed Chapters 42–56 and reviewed/corrected every chapter, supported by GPT-5.6 Luna High research. The lead independently reviewed the material and integrated the checks below. Detailed chapter findings are in `astra-review-*.md` and `REVIEW_LOG.md`.

## Course-wide checks

- All 56 reference projects pass formatting, Clippy with warnings denied, and ordinary tests. CPU default demonstrations pass; GPU execution is checked separately below.
- All 56 learner projects pass formatting, Clippy, and their useful starting run. Completed learner work is accepted. Remaining incomplete tests fail at the intended TODO/repair, rather than at compilation. All 56 Rustlings solutions pass and unfinished exercises fail intentionally.
- Before learner edits, the official Rustlings 6.5.0 `dev check` passed the complete 56-exercise community pack, including its formatting and solution checks. Its authoring mode expects unfinished exercises; the consistency revision preserves the learner’s completed Chapter 1 exercise. Current-run handling is recorded separately below.
- The static content audit finds no missing internal resources or anchors, duplicate chapter IDs, or unrendered LaTeX. All 56 pages have substantive lessons and progressive answer disclosures. Reports: `validation/content.json`, `validation/ch01.json` through `ch56.json`, and `validation/rustlings-final.txt`.
- Browser inspection visited all 56 chapter pages: correct titles, lesson sections, loaded diagrams, and no horizontal page overflow. Chapter 55/56 screenshots were inspected; narrow 390×844 layout, dark/light themes, mobile navigation and keyboard-operated disclosures passed. Keyboard glossary focus creates its description and Escape dismisses it. Search finds chapters and terms; source links display readable Rust and copying places the displayed source on the clipboard.
- Earlier interactive checks reproduced Chapter 1's first update (loss 9 → 3.52) and learned weight/bias (2,1); Chapter 35's causal mask excludes the future and its unmasked variation includes it. Reading progress survived reload. `node tools/test_glossary.cjs` passes actual touch/focus/Escape/mouse event-handler checks; no physical touchscreen test was performed. Website operation itself needs no Node package or remote assets.

## Actual Radeon execution

Host: Ryzen 9 9900X and Radeon RX 6950 XT, RADV NAVI21, Mesa 26.2.2-arch3.2, Vulkan 1.4.354. The initial missing device was sandbox isolation; approved host execution resolved access. Kernels use wgpu 29.0.4 and original WGSL.

- Chapter 29: 67-element vector addition, including the dispatch tail.
- Chapter 30: tiled products with shapes 3×17×5 and 17×19×33; 777-element, two-stage reduction, including the final GPU reduction.
- Chapter 31: device-resident 2–4–1 tanh XOR training, 800 steps, loss 0.730555 → 0.013002; 80-step CPU parameter comparison.
- Chapter 32: separate, fused and supported f16 kernels agree with the CPU reference for every one of seven timing samples. Timestamp queries and shader f16 are supported on this device. No universal speedup is claimed.
- Chapter 36: after the final numerical corrections, a two-layer odd-shaped decoder matches CPU logits, every gradient and three updates within its documented tolerances.

Passing evidence: `validation/gpu-final-runs.json`, `gpu-ch30-staged.json`, `gpu-ch32-all-samples.json`, and `final-gpu-decoder.json`. The older `gpu-hardware.json` retains initial history, including the corrected WGSL reserved-word failure.

## Data and training checks

All four MNIST archives passed their published checksums; provenance and derived split rows are recorded locally. The fit/validation split is 55,000/5,000; the official 10,000-row test set was integrity-checked but **not scored**.

| Chapter | Bounded run | Validation result |
|---|---|---|
| 10 | Linear model, one epoch, 55k/5k | loss 2.3026 → 0.3523; accuracy 10.0% → 90.1% |
| 11 | MLP, one epoch, 55k/5k | loss 2.4401 → 0.4512; accuracy 10.9% → 88.0%; checkpoint step 1719 |
| 20 | CNN, one epoch, 2k/2k development rows | loss 2.3018 → 0.4614; accuracy 10.2% → 86.7% |

These runs verify real-data paths, not a tuned or controlled model comparison. A Luna reviewer repeated the existing Chapter 20 local smoke command during final review, reproducing its result; no new download or official-test evaluation occurred.

Chapter 39's exact **14,442,496-parameter** configuration completed full backward and AdamW updates on both CPU and GPU-GEMM paths; CPU checkpoint continuation and tiny GPU train/save/resume passed. Persistent model plus Adam storage is 173,309,952 bytes, excluding gradients and caches. Large checks used two eight-token microbatches and sixteen held-out targets. Observed single-update rates were about 74–76 tokens/s CPU and 57 tokens/s GPU bridge; these short checks are **not steady-state throughput estimates**. The bridge transfers data per operation and can be slower. Evidence: `validation/lm-large.json`, `lm-device-cli.json`, `lm-large-gpu.json`.

**Extended language-model training was not run.** No general-purpose language quality, large sparse-model speedup, or multi-GPU execution is claimed. Distributed experiments use local miniature execution, with their limits taught explicitly.

## Framework and serving checks

- Chapter 55: optional pinned Burn 0.21.0 CPU integration passes formatting, strict Clippy, real forward/loss/gradient/SGD assertions, weight-and-bias recorder restoration, and actual scratch export → Burn import on three probes. Evidence: `validation/final-framework.json`.
- Chapter 53: actual local train → artifact → server requests return 200 for valid data and 400 for invalid data; accepted requests update the live monitor. The host loopback test also passes.
- Chapter 56: the contextual sparse decoder trains, saves, resumes, generates, and loads its artifact into a real single-request HTTP server. Valid/invalid requests return 200/400. Full small-model derivatives, independent dense equivalence, causal inference, routing/capacity, malformed checkpoints and exact continuation pass. Evidence: `validation/final-serving.json` and the chapter's test report. This is a tiny ASCII language-model experiment, not evidence of useful general language ability.

The course runs locally. Reading assets are bundled; initial toolchains, dependencies, optional datasets and external reference pages need connectivity. No external hosting or long-running training was performed.

## Reader and workflow update

Added explicit syntax labels throughout all 56 HTML lessons, locally bundled Prism 1.30.0 and its MIT license, themed token colors, and source-page highlighting. Plain terminal output remains uncolored. Reader asset URLs carry content hashes so an existing open course reloads updated styles/scripts. `node tools/test_highlighting.cjs` checks six grammars, Rust lifetimes/raw strings/macros and exact source preservation; `node tools/test_glossary.cjs` still passes. Browser inspection confirms Rust/WGSL/TOML/Python highlighting, light/dark rendering, and actual copy-to-paste preservation.

The original reader/workflow update added a justfile that delegates to existing Cargo/Rustlings/course tools through a small argv-only chapter adapter. Installed just 1.58.0 accepts its syntax and formatting. Real `just run 1`, `just test 01`, starter launch, intended starter-test failure, formatting check, exercise help, MNIST self-check and `just verify 1` pass. Subdirectory use, invalid chapter rejection and literal metacharacter/space forwarding were checked; downloads and GPU execution remain explicit. No model arithmetic changed.

The new HTML Git guide covers initial baselines, staged learning checkpoints, notes, date/history inspection, experiments/recovery and GitHub remotes/push. Official Git/GitHub sources were verified with Luna High research; the lead reviewed the final instructions. It is linked in desktop/mobile navigation, setup, footer and local search. Its shell examples passed syntax checks without running Git mutations. Browser checks show no horizontal page overflow at 390×844; code blocks scroll independently. No repository initialization, commit or push was performed for this update.

## Helix project discovery

Installed the selected stable toolchain's rust-analyzer and rust-src components and recorded them in `rust-toolchain.toml`. `.helix/languages.toml` selects that root marker and explicitly links all 114 independent Cargo manifests, preserving user-level options. Helix 25.07.1 accepts the configuration; rust-analyzer 1.96.0 loads 114 workspaces. Actual LSP checks return Chapter 1's `weight: f64` hover, the Chapter 56 starter's definition in its reference library, and `Vec`'s definition in the installed standard-library sources. Initial missing analysis dependencies were fetched during the authorized host check. Evidence: `validation/rust-analyzer.json`. An already-open Helix session must reopen the workspace to read the new project configuration.

## Consistency revision: all 56 chapters

At the learner’s request, each chapter received its own GPT-5.6 Sol High author, working in prerequisite order with at most four chapter authors active. Luna High agents supported terminology, primary-source and interface checks. The lead read every chapter’s revised lesson, reference implementation, starter/exercise mapping and review report, coordinated downstream API changes, and corrected remaining code/prose mismatches. Individual records are `consistency/ch01.md` through `ch56.md`; the shared contract is `CONSISTENCY.md`, with an HTML reader guide at `/conventions.html`.

Chapters 1 and 2 now share the same `Neuron`, data, `predict`, `loss`, `numerical_gradient`, `step`, and `train` interfaces. Their shared method bodies agree; only the gradient called by `step` changes. Chapter 6 retains the same prediction and ordinary MSE. Later chapters explain changes in model shape, ownership, precision, loss reduction and optimizer responsibility before using them. Fused parameter derivatives use `loss_and_gradient`; the separate logits primitive explicitly names its returned logit gradient. Glossary ownership and genuine specializations are documented instead of competing definitions.

Final integrated checks pass across all 56 reference and learner projects: formatting, warning-denied Clippy, reference tests, bounded CPU demonstrations, starter launch, solution tests, and intended learner behavior. The learner’s completed Chapter 1 exercise and 1,000-step starter experiment remain completed; Chapter 2’s added helper experiments remain present. Unfinished tasks fail at their intended TODO/repair. The official Rustlings authoring check rejects the original pack because Chapter 1 is already solved. It passes an isolated copy with only that exercise marked `skip_check_unsolved`; the real pack and learner progress were not changed. Evidence: `consistency/rustlings.json` and refreshed per-chapter validation JSON.

`just consistency` passes: every chapter has a continuity section and code-guide link, at least one exact source-checked implementation excerpt, a chapter review record, unique glossary ownership, and no old fused-gradient aliases. Tagged excerpts match their defining source after whitespace normalization. The full website audit has no missing links or anchors, duplicate IDs, placeholders, or unrendered LaTeX. Browser checks cover every final chapter at desktop and 390-pixel width, including loaded diagrams, highlighted code, continuity sections, and horizontal overflow. The shared guide, search and keyboard terminology were also checked. The local highlighting and glossary event-handler tests pass.

The rewritten GPU paths were executed on the Radeon RX 6950 XT: Chapter 29’s partial workgroup, Chapter 30’s odd-shaped products and staged reduction, Chapter 31’s explicit-data training and CPU parity (including a different three-row dataset), Chapter 32’s separate/fused/f16 comparisons, Chapter 36’s complete decoder gradients and updates, and Chapter 39’s tiny GPU save/resume path. Evidence: `consistency/hardware.json`. The original extended-shape and real-MNIST evidence above remains historical; this consistency pass did not repeat extended training or claim new language quality or speedup.

Fresh Chapter 55 exports are byte-identical to the pre-revision artifact and import into Burn on three probes. Chapter 56's two-step checkpoint is byte-identical to its pre-revision artifact; that old artifact resumes successfully under the renamed interfaces. Actual Chapter 53 and 56 local server requests return 200/400 for valid/invalid inputs, with the Chapter 53 monitoring/rollback test passing. Evidence: `consistency/framework.json` and `consistency/serving.json`. No commit, push, external deployment or extended training was performed.

## Final section review

After the consistency revision was committed and pushed, nine GPT-6 Astra Medium reviewers each checked one complete course section. Six sections required no changes. Focused fixes address Chapter 11 checkpoint temporary-file collisions, Chapter 19 empty folds with small classes, and Chapter 39 incompatible vocabulary or implicit large-model resume settings. Lessons 10, 11, 18, 19, and 39 received corresponding mathematical or continuation clarifications; small issues were left untouched.

The lead reviewed every fix, reran the three affected chapter gates, rebuilt and audited the complete site, checked the edited pages at desktop and phone widths, and confirmed `just consistency` has zero findings. Bounded GPU tests and a tiny GPU save/resume passed. Learner work was unchanged. Coverage, individual section reports, exact checks, and limitations are recorded in [the section review](section-review/README.md).
