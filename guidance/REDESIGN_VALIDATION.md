# Redesign validation

Validation date: 2026-09-09. The complete redesigned course is ready for private publication. The deployment is performed separately from this source validation record.

## Delivered scope and preservation

- All 56 chapters retain the original topics and nine section boundaries. Their metadata maps every topic to an authored learning step. The final course contains 426 steps and 152 approximate sessions.
- Nine independent Rust lab packages provide 56 working learner baselines, separate completed algorithms, inspectable learning-goal checks, progressive hints, and controlled transfer experiments. The nine design reports in [redesign/](redesign/) form the chapter-by-chapter design matrix.
- Exactly 18 focused browser tools match the approved catalog. The source download includes the new labs, complete reference projects, local site, tooling, source records and datasets; it excludes old starters, the retired exercise runner, build caches and temporary authoring files.
- The branch is `codex/ml-learning-redesign`. `main` and the original `projects/` and `exercises/` files remain unchanged. The build writes only generated output and never rewrites learner sources, the chapter map or old exercise configuration.

## Rust and scientific checks

The lead integration gate passed for all nine packages: formatting, strict Clippy, unit/integration tests, every default baseline, every completed-solution goal check, and every delivered learner-goal status. There are 172 passing Rust tests in these package runs; two hardware/socket tests remain opt-in rather than part of the ordinary suite. Related GPU and serving paths were exercised explicitly as described below. Full command output is retained in `validation-redesign/section-01.json` through `section-09.json`.

An intentional unfinished algorithm reports `GOAL_NOT_MET:`. Numerical equality that cannot establish a structural implementation or hardware claim reports `GOAL_REVIEW_REQUIRED:` with visible review criteria. Ordinary runtime/input failures cannot count as acceptable unfinished work. The shared regression check covers these distinctions.

Real GPU solution checks for Chapters 29–32 ran on the available AMD Radeon RX 6950 XT using RADV/Vulkan. Chapter 31 reached BCE 0.013002 after 800 steps, with maximum parameter error 2.861×10⁻⁶ against the CPU path. These are fixture-specific correctness observations, not general speed or quality claims. See [hardware.json](validation-redesign/hardware.json) and the Section 06 report.

The language-model capstones include actual training, evaluation, serialization and resumed-update evidence. Chapter 56 also serves a real bounded loopback generation request. Its resumed 180-step checkpoint matches uninterrupted training byte for byte. The tiny held-out loss worsened after additional fitting; that result is preserved. Chapter 19 freezes model selection before its explicit `--final` report, retains errors and uncertainty, and never silently reselects a model from final results.

Optional external datasets, larger runs and the preserved Chapter 55 Burn integration are outside the default checks. Burn was not executed; the independently implemented runtime, actual file interchange and forward/gradient/update parity were checked. Browser illustrations never claim to run these Rust or hardware paths.

## Reader and browser verification

- The generated site passes the full 56-chapter metadata/content/source-excerpt/local-link/anchor audit with zero findings. All nine section pages and the 243-term glossary build successfully.
- Every chapter was opened in a real browser at 1366×900. Focused mode displays one step; Full chapter displays exactly the authored count. All 56 have no document-level horizontal overflow.
- Every chapter was checked in full view at 390×844. The routing table's narrow-screen overflow was fixed using the existing table scroller and rechecked. All 56 fit the viewport; closed mobile navigation is inert. Keyboard opening, search focus, Escape, and return focus were checked.
- Search returns step-specific links and opens the corresponding visible step. Query and nested fragment deep links work. Previous/next controls preserve the selected step. Homepage Continue restores the reading position.
- Practice checkboxes and a temporary note persisted after reload; test entries were then cleared. The interface explicitly labels these as self-reports, independently of reading position or demo use. Light and dark themes were visually checked.
- With scripts removed from a temporary generated chapter, all its steps remained readable and JavaScript-only reader controls stayed hidden. Each interaction also retains static calculations or a readable worked alternative.

All 18 tools were exercised with real browser controls. Representative observed results include:

| Tool | Changed case and observed result |
|---|---|
| Sensor fitting / gradients / classification | Zero parameters give MSE 9; the simultaneous rate-0.1 update gives 3.52; no positive predictions produces undefined precision. |
| XOR / autodiff / tensors | Hidden features separate the four XOR corners; the shared `x²+x` paths sum to derivative 7 at x=3; doubled dense inputs give `[-3.5,7.5,-3.5,25.5]`. |
| Sampling / leakage / geometry | Seeded machine bootstrap gives `[0.5000,0.8333]`; confidence multiplier 3 gives Brier 0.2167; the strict split has 48/18 rows and no shared machines; moving center 1 to x=2.5 changes the selected center. |
| Convolution / embeddings | The first response is 1.2 and the first pool winner is 1.8; changing stride alters the response map. Scaling C to `[0,3]` changes its vertical-query dot product to 3 while cosine remains 1. |
| GEMM / threads / GPU | Completed GEMM gives `[30,36,42,66,81,96]`; 19 values at lane width 8 leave tail 3; N=67 launches 128 lanes with 61 guarded lanes. |
| Tokens / attention | One BPE merge changes 11 byte tokens to 9; masked future probability is zero despite score 100, and unmasking changes the weighted value to `[20,3]`. |
| Inference / routing | Twelve prefix positions use 1536 f32 cache bytes; weight precision leaves this cache unchanged. Collapsed routing admits 4/12 at capacity factor 1 and 8/12 at factor 2, preserving the attempted-route balance value. |

The separate [interactive review](validation-redesign/interactives.md) records the exact 18 IDs, nine dependency-free numerical commands, syntax checks, all corrected findings, labels and accessibility equivalents. Shared glossary, navigation and six-language syntax-highlighting checks also pass.

## Independent review and downloadable copy

Three independent substantive reviews cover [Chapters 01–19](validation-redesign/review-01-19.md), [20–39](validation-redesign/review-20-39.md), and [40–56](validation-redesign/review-40-56.md). They trace actual learner implementations into their checks, review prerequisite order and worked reasoning, and record each material finding and resolution. No material P1/P2 remains after the final fixes.

The complete source ZIP was extracted into a separate temporary directory. All nine Cargo manifests resolve their local dependencies inside that copy. The copy builds its own site and runs the first and tabular-capstone labs, including its explicit completed final-report command. [download.json](validation-redesign/download.json) retains the clean-copy observations. This verifies packaging and relative paths; a fresh computer still needs the documented Rust/Python tools and a one-time dependency fetch where applicable.

`python3 tools/test_course.py`, `node tools/test_glossary.cjs`, `node tools/test_highlighting.cjs`, Python syntax checks and `git diff --check` pass. Validation establishes the delivered behavior and inspectable numerical evidence; written explanations, experimental judgment and optional larger/hardware work retain their stated human assessment boundaries.
