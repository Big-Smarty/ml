# Redesign authoring contract

This is the active contract for `codex/ml-learning-redesign`. The approved September 2026 plan supersedes older Rustlings, per-chapter authorship, starter failure, consistency-revision and minimum-word-count requirements. Historical review files describe the earlier course, not this redesign.

## Audience and learning

Everyday Rust programmer; complete ML beginner. Keep all assigned topics in course.json, its order and nine bundles. Introduce math just in time with actual inputs, intermediate arithmetic, defined notation and shapes, then small corresponding Rust excerpts. Explain why a method is useful and a case where it fails. Write connected teaching prose, not outlines or repetitions of a six-item template. Chapters may require several 30–45-minute sessions. Each session recalls relevant knowledge, explains a worked example, asks a prediction, runs a working baseline, asks for meaningful algorithm implementation, compares evidence and transfers to unfamiliar inputs. Include progressive hints and fully explained answers in native details. No navigation gates. Never imply baseline success or clicking a diagram proves mastery.

## Ownership

One GPT-6 Astra High owner per full existing section, not per chapter. Read guidance/ASSIGNMENTS.md and CHAPTER_TEMPLATE.md. Owners edit only their chapter directories, their one labs package, and guidance/redesign/section-NN.md. Read existing lesson/meta/research/project code completely before replacing instruction. Original projects/chNN are reusable reference algorithms and preserved learner history; request lead help before changing them. Root owns shared site/build/tooling/course.json, sections.json, README, editor configuration, hosting and integration. Do not commit, push, publish, install a framework or mutate another owner's files. Research uses GPT-5.6 Sol High. No need for further research delegation when existing evidence settles a point.

## Authored content and metadata

lesson.html is a semantic HTML fragment, no h1 or document shell. Wrap each ordered reader step in `<section class="learning-step" id="stable-slug">`, beginning with h2. Existing meaningful anchors may appear on nested headings. Each step should address one coherent question, not merely one short paragraph. Include enough explanation to make all topics comprehensible. Native paragraphs, tables, pre/code, details/summary; approved classes include lead, callout, warning, equation, worked-example, exercise, table-wrap, caption. Escape source snippets. Code language classes: language-rust, language-wgsl, language-bash, language-toml, language-none. Source links use `/code/labs/SECTION/src/chNN.rs` or preserved `/code/projects/chNN/...`; guide routes `/chapters/NN.html?step=SLUG`, glossary `/glossary/SLUG.html`.

Retain outcomes, terms (slug -> name/definition/explanation), sources (title/url/note), checks, limitations in meta.json. Add:
```
"redesign_version": 2,
"steps": [{"id":"problem", "title":"What is the sensor getting wrong?", "session":1, "minutes":8}],
"sessions": [{"id":1, "title":"Fit and inspect a predictor", "goal":"Explain each error and implement a complete update"}],
"lab": {"package":"s01-foundations", "chapter":"01", "entry":"labs/s01-foundations/src/ch01.rs", "goal":"Train on calibration samples and verify unseen inputs", "check":"just lab-check 01"},
"interactives": [{"id":"neuron-fit", "title":"Fit a sensor", "file":"demo.js", "anchor":"neuron-fit", "illustration":true}],
"topic_coverage": {"Prediction":"problem", "numerical slopes":"train"}
```
All topic_coverage keys must match course.json topics verbatim; their values name actual step ids. steps ordered exactly like lesson DOM. Sum minutes per session approximately 30–45; short chapter can have one session and long chapter many. Do not silently mark difficult original topics optional. Meta `lessons` may mirror step titles. Each step must have an explicit observable goal or question; each session must include implementation and transfer somewhere. Research.md records original/source verified claims, limits and author route. Avoid fabricated outcomes or experiments.

## New lab contract

Nine independent Cargo packages under labs/ (see assignments). Stable Rust edition 2021 or 2024; `[workspace]` per package; stdlib and existing dependencies first. One learner `src/chNN.rs` and separate explained `src/solutions/chNN.rs` per chapter, shared plumbing only as needed within package. Retain core reference algorithm correctness when reusing projects libraries. Each package CLI:
- `cargo run --manifest-path labs/PACKAGE/Cargo.toml -- NN` runs the learner's useful, bounded baseline and prints inputs/results. Extra experiment arguments allowed after NN.
- Same with `--check` invokes the learner's substantial learning-goal checks and returns nonzero with evidence when goal isn't met.
- Same with `--solution` runs the separate completed solution; `--solution --check` verifies it.
- `cargo test` checks supplied baseline, plumbing, shapes/error boundaries and solution numerics, and must pass on delivery. It must NOT pretend unfinished learning goals pass. No intentional startup panic, todo! or watch runner. Working baselines may implement an earlier simpler model; explain the algorithm the learner will build/replace. Do not recreate missing-one-expression tasks.

Typical module interface: `pub fn run(args: &[String]) -> Result<(), String>` and `pub fn check() -> Result<(), String>`. Keep CLI clear and dependency-free. If a different internal organization materially helps, preserve the exact public CLI and source entry mapping. `just lab NN` and `just lab-check NN` are supplied by root. Tests are readable locally and call actual learner functions; explanations distinguish automated numerical evidence from self-assessed prose. Include at least one unfamiliar check input, justified tolerances, one meaningful failure/limitation, and an independent variation per chapter. Separate solutions teach the reasoning, not just code. README in each lab explains commands, actual scope, checkpoints and datasets. Supplied parsers, RNG, I/O, shape validation remain complete. All default computations are tiny/offline. Downloading, extended training and real GPU execution require explicit flags. Hardware absence must be an explicit unsupported result, not a fabricated pass. GPU chapters include actual WGSL dispatch and CPU agreement, not only diagrams. Capstones must actually train/evaluate/serve stated architectures.

## Interactives

Implement only the 18 assigned tools in ASSIGNMENTS.md. A reusable tool may be placed in its first chapter and linked from later relevant chapters; do not copy scripts. Each demo uses chapter-local demo.js loaded automatically when present. Scope query selection to its unique container; no external scripts/network. Explain labels, ask prediction before changes, use labelled native inputs/buttons, deterministic Reset, text equivalent values and static worked example with no JS. Describe assumptions and that calculations illustrate rather than execute Rust. Keyboard alternatives, no color-only information, SVG title/desc, accessible outputs. Vary one relevant factor at a time. No giant dashboard of unrelated topics. Record deterministic example checks in the section report (automated pure calculation checks welcome). Do not attach site progress to demo completion.

## Validation and final handoff

Read guidance/NUMERICS.md; redesign contract wins if legacy convention conflicts. Apply Rust skills. Run fmt --check, clippy --all-targets -- -D warnings, cargo test, each chapter baseline, each solution --check, inspect expected nonzero learner goal checks. Do not run every unrelated package. Report actual output summaries and limitations. Document each chapter in guidance/redesign/section-NN.md: original topics; objectives; explanation/step mapping; meaningful learner change; check and transfer; session count; interactive values; tests run; prerequisite audit. This is the 56-chapter design matrix. Ask root early about cross-section references/interfaces. Never declare completion when a promised topic or capstone is only a placeholder.

## Delivery status and clean handoff (integration clarification)

Learning-goal checks must distinguish an expected unmet algorithm goal from an unexpected runtime/infrastructure error. Prefix **only intentional goal-comparison failures** with `GOAL_NOT_MET:` in their returned error/output; exit1 is conventional. Do not add that prefix indiscriminately to every error from a check: missing fixtures, invalid CLI options, corrupted checkpoints and unexpected device/setup failures remain ordinary errors. Successful goal checks exit0. Where numerical equality cannot assess a structural learning goal (for example a correct serial GPU baseline versus a cooperative kernel), report `GOAL_REVIEW_REQUIRED:` with exit3 and a concrete source/trace rubric instead of claiming automatic completion. Record that manual criterion in metadata `lab.manual_checks` and the lesson. Completed solutions must still pass their stated automatic checks, with author source/hardware review documented separately. The lead accepts only explicit expected-goal/manual-review statuses in delivery verification.

Remove temporary authoring generators and old-lesson backup copies from your lab before handoff. Use /tmp for scratch work. The downloadable course should contain intentional learner/runtime/source/data/reference files, not original-NN.html, old-NN.html, one-off make/extend scripts, test output or temporary checkpoints. Keep useful reproducible dataset generators when the lesson actually uses them.
