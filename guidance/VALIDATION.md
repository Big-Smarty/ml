# V2 validation requirements and status index

**Status:** the redesigned implementation is integrated and privately published. The recorded course, Rust, and available-browser checks pass. The full accessibility validation target below is not yet complete: actual browser zoom and target assistive-technology checks remain pending, as detailed in [VISUAL_STYLE_VALIDATION.md](VISUAL_STYLE_VALIDATION.md).

The initial redesign checks are recorded in [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md); the subsequent theme, MathML, command, and browser checks and their limitations are recorded in [VISUAL_STYLE_VALIDATION.md](VISUAL_STYLE_VALIDATION.md). Section-scoped evidence belongs in the owner-authored `guidance/redesign/section-NN.md` matrices. The completed pre-v2 validation record from 2026-09-08 is preserved at [archive/2026-09-08-validation-record.md](archive/2026-09-08-validation-record.md) and does not satisfy v2 gates unless a current check is explicitly rerun and recorded.

Review criteria are in [REVIEW.md](REVIEW.md). Authoring and status conventions are in [AUTHORING.md](AUTHORING.md) and [CONSISTENCY.md](CONSISTENCY.md).

## Required delivery evidence

The redesign is complete only when current evidence establishes all of the following:

### Curriculum and content

- All 56 chapters have `redesign_version: 2`, valid ordered learning steps, session metadata, topic coverage, lab mapping, checks, sources, and limitations.
- Every `course.json` topic is substantively taught and maps to a real step; difficult topics are not silently made optional.
- Every chapter has a runnable baseline, worked numerical or state trace, meaningful learner implementation, diagnosis or controlled variation, progressive hints, explained answer, and unfamiliar transfer case.
- Chapters are divided into coherent 30–45-minute sessions as needed while retaining one continuous full-chapter view.
- The nine `guidance/redesign/section-NN.md` reports contain a complete 56-row matrix, commands, results, limitations, and cross-section handoffs.
- Internal links, step anchors, glossary links, source-code links, terms, and prerequisites resolve without duplicates or missing targets.
- Source notes support their adjacent claims and preserve limitations and counterevidence.

### Nine cumulative labs

- All nine lab packages exist at the paths in [ASSIGNMENTS.md](ASSIGNMENTS.md) and implement the CLI contract in [AUTHORING.md](AUTHORING.md).
- Every default learner baseline runs offline and prints a useful bounded result.
- Every learner goal check calls the learner's real implementation and distinguishes expected `GOAL_NOT_MET:`, documented `GOAL_REVIEW_REQUIRED:`, unexpected errors, and success.
- Every completed solution and solution check passes.
- Package formatting, warning-denied Clippy, tests, chapter baselines, solution checks, expected learner-goal states, and manual rubrics are recorded.
- Checks use readable local cases, meaningful boundaries, justified tolerances, and the actual section-project interfaces.
- Capstones actually train, evaluate, resume, or serve the architecture promised at the documented tiny scale.
- Defaults avoid network access, long training, and unavailable hardware; optional paths are explicit.

### Reader and interactions

- The exact 18 full interactions assigned in [ASSIGNMENTS.md](ASSIGNMENTS.md) exist and pass their documented deterministic fixture checks.
- Every interaction implements Question → Predict → Observe → Explain → Build → optional Explore without gating controls or navigation.
- Browser arithmetic matches the lesson, local solution, and section report; simulated values are not labeled as Rust or hardware execution.
- Every other chapter includes a useful lightweight retrieval, prediction, trace, or diagnosis activity.
- Focused-step and continuous modes use the same semantic content and stable URLs; direct links, Back/Next, search, glossary, progress, and reset work.
- With JavaScript unavailable, continuous lessons, answers, commands, source links, navigation, glossary, and static teaching equivalents remain usable.
- Keyboard-only operation, non-drag input, visible focus, text equivalents, non-color cues, reduced motion, status messages, 320-pixel reflow, 400% zoom, light/dark themes, and print/static output are checked.
- Progress states remain separate, literal, local, self-reported, and free of mastery claims.

### Numerical, data, systems, and hardware evidence

- Deterministic oracles and tolerances validate key forward, loss, gradient, update, shape, serialization, and resume paths.
- Stochastic claims use stated seeds or repeated evidence; generalization claims use proper splits, baselines, and held-out evaluation.
- Data preprocessing is fitted on training data; grouped/time boundaries, prediction-time availability, duplicate/leakage checks, and selection/final-test separation are exercised.
- CPU optimization results preserve correctness and record local machine/toolchain/configuration, warmup, samples, statistic, and dispersion.
- GPU conceptual, compilation, actual-dispatch correctness, and performance states remain separate. Hardware verification requires adapter, dispatch, readback, and parity; performance additionally requires actual timing and metadata.
- Checkpoint and cross-section artifacts preserve model, optimizer, progress, RNG, vocabulary, preprocessing, and format state where applicable.
- No short smoke run is reported as steady-state throughput, model quality, convergence, or universal speedup.

### Integration and publication

- Shared build, content audit, course dispatcher, lab mapping, source archive exclusions, and learner-file preservation checks pass against the final tree.
- Cross-section APIs, shapes, layouts, reductions, artifacts, and links are exercised at their consumers.
- The complete generated site is inspected through all 56 chapters and nine section views, including representative desktop and phone layouts.
- The final source commit is packaged and deployed to the configured private OpenAI Pages site.
- The deployed site reports the reviewed content, remains private, and is verified by URL after deployment.
- Final delivery names remaining optional, hardware-dependent, or extended-training work without presenting it as complete.

## Evidence scope and status language

Validation statements name the exact command, files, environment, outcome, and limitation. Use:

- **authored** when files exist;
- **owner-checked** when named section-scoped checks passed;
- **integrated** only after cross-section and full-course gates pass;
- **hardware-verified** only after the named path ran on reported hardware;
- **deployed** only after the reviewed source is live and visibility is checked;
- **pending** when work or evidence remains.

A check inherited from the pre-v2 course remains historical until rerun. A conceptual fallback does not satisfy a hardware gate. A browser check does not satisfy a local Rust check. A successful solution does not show that the learner path or transfer task is valid.

## Current record

[REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md) and [VISUAL_STYLE_VALIDATION.md](VISUAL_STYLE_VALIDATION.md) record checks already observed and checks still pending. Update the relevant record with evidence as work completes; do not replace pending items with broad claims or copy pre-v2 results forward. A 320-pixel reflow check does not establish that actual 400% browser zoom was tested.
