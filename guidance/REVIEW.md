# V2 section and course review contract

This file defines review requirements for the active learning redesign. It is not a validation record and does not claim that the 56 chapters, nine labs, 18 interactions, capstones, or private deployment are complete.

Read [AUTHORING.md](AUTHORING.md), [ASSIGNMENTS.md](ASSIGNMENTS.md), [CONSISTENCY.md](CONSISTENCY.md), and [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md) before review. Record observed integrated results and remaining gates in [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md). Pre-v2 review records are archived in [archive/README.md](archive/README.md).

## Review levels

1. **Owner review:** each section owner checks every assigned lesson, metadata record, lab module, solution, interaction, and matrix row before handoff.
2. **Cross-section review:** the lead checks prerequisites, artifact interfaces, terminology, data formats, numerical conventions, and links across section boundaries.
3. **Integrated course review:** the lead runs the complete content, lab, reader, accessibility, source, and publishing gates after all owner work lands.
4. **Deployment review:** the lead verifies that the deployed private site corresponds to the reviewed source commit and reports the actual URL and visibility.

Passing an owner check is not integrated-course completion. A section report may state exactly what it verified while leaving shared-site, hardware, or deployment gates pending.

## Chapter and session review

For every chapter, verify that:

- all assigned `course.json` topics appear substantively and map to real step IDs in metadata;
- the full chapter and guided steps use the same semantic content and stable anchors;
- broad chapters are split into coherent 30–45-minute sessions without dropping difficult topics;
- each session recalls a relevant dependency, explains one worked case, and reaches an observable stopping point;
- symbols, shapes, units, reductions, and technical terms are defined before use;
- concrete values, diagrams or traces, notation, and Rust explicitly correspond;
- a prediction is interpretable and reconciled with evidence rather than used as an access gate;
- at least one limitation or failure case is explained;
- retrieval and transfer prompts include worked answers or self-check criteria;
- prose distinguishes external research results, local measurements, analytical models, and design judgment.

## Lab and assessment review

For every v2 chapter lab, verify that:

- the default baseline runs offline and produces a meaningful result before learner edits;
- the learner owns a coherent ML or systems mechanism rather than incidental plumbing or one missing expression;
- supplied parsers, allocation, CLI, serialization, device setup, HTTP, and similar infrastructure match the chapter's stated ownership boundary;
- the learner goal check calls the learner's real public entry point and fails only with the documented `GOAL_NOT_MET:` condition when unfinished;
- completed solutions pass their documented automatic checks;
- `GOAL_REVIEW_REQUIRED:` is used only when numerical automation cannot validly establish the structural goal and the manual rubric is recorded;
- checks cover several values, shapes, seeds, or boundaries where appropriate and use justified tolerances;
- at least one unfamiliar but inspectable transfer case differs from the worked example;
- stochastic, generalization, or performance claims use suitable repetitions, baselines, splits, and environment metadata;
- free-text reasoning remains honestly self-assessed rather than fake-autograded;
- optional downloads, extended training, real GPU execution, and architecture-specific paths are explicit rather than default prerequisites.

Do not review against Rustlings, old failing starters, per-chapter project skeletons, hidden tests, or historical Neuron signatures. Those belong to the archived course workflow.

## Interaction and reader review

For each of the 18 assigned full interactions, verify:

- one clear learning objective and a Question → Predict → Observe → Explain → Build → optional Explore sequence;
- all controls remain available without navigation or answer gates;
- one relevant variable changes at a time and held-constant values, units, assumptions, and Reset are visible;
- fixture arithmetic matches the lesson, local solution, and section report;
- the browser calls itself an illustration and never claims to execute Rust or measure hardware;
- keyboard operation, non-drag alternatives, visible focus, non-color cues, text/table equivalents, and a useful static/no-JavaScript path;
- no autoplay; step, pause, reset, and reduced-motion behavior where temporal change matters;
- usable reflow at 320 CSS pixels and 400% zoom, with any necessary two-dimensional visual contained and labeled;
- a failed script leaves the core explanation, answer, command, and code link available.

For the reader as a whole, inspect focused-step and continuous modes, direct step links, Back/Next, search results, section navigation, glossary behavior, progress labels, print/static rendering, light/dark themes, keyboard focus, and localStorage reset. Progress wording must remain literal and self-reported.

## Scientific and source review

- Use original papers, official documentation, standards, and original project sources for technical claims.
- Bind reported numbers to the cited model, dataset, hardware, software, and measurement.
- Preserve negative results, boundary conditions, and uncertainty.
- Do not convert a curriculum precedent, visualization design account, artifact framework, or popularity measure into causal learning evidence.
- Do not claim that the complete redesign, Rust, projects, productive failure, concrete-first order, or one retrieval schedule is empirically superior.
- Check all source URLs and ensure each note says what the source supports and what it does not.

## Integration and project review

For each of the nine section projects, confirm:

- chapter artifacts accumulate through a stable interface or explicitly documented conversion;
- project work integrates already taught concepts;
- learner responsibility fades across the section without a blank-repository jump;
- required core work and optional extensions are distinct;
- the stop condition includes core evidence, one design or failure explanation, and a held-out or transferred case;
- provided checkpoint states let a learner rejoin without completing every earlier optional task;
- the section report contains every chapter row, exact commands, results, limitations, and downstream handoffs.

For cross-section dependencies, verify tensor layouts, vocabulary/token IDs, checkpoint state, data splits, preprocessing state, precision, optimizer state, RNG state, device behavior, and file formats at the actual consumer.

## Status language

Use only evidence-scoped statements:

- **authored:** required files exist, without implying checks passed;
- **owner-checked:** named scoped checks passed in the section;
- **integrated:** cross-section and full-course checks passed;
- **hardware-verified:** the named kernel or path executed on reported hardware and produced the reported evidence;
- **deployed:** the reviewed source commit is live at the private site and visibility was checked;
- **pending/blocked:** name the exact remaining gate and evidence needed.

Never report a conceptual fallback as hardware verification, a short check as steady-state throughput, or a successful tiny model as general language quality.

## Required records

- `guidance/redesign/section-NN.md`: owner-authored 56-chapter matrix and scoped evidence.
- [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md): current integrated observations and remaining gates.
- Build/audit output paths named by that validation log.
- Private deployment source commit, URL, and visibility check at final delivery.

Historical `astra-review-*`, `consistency/*`, `section-review/*`, and earlier validation JSON may remain useful provenance for the pre-v2 course. They do not satisfy a v2 gate unless the current validation explicitly reruns and records the corresponding check against v2 files.
