# V2 research and source discipline

This is the active research contract for the September 2026 redesign. The full learning-science synthesis, counterevidence, curriculum comparison, transfer analysis, and assessment-validity rationale are in [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md). Current implementation ownership is in [ASSIGNMENTS.md](ASSIGNMENTS.md), and chapter-level technical research belongs in the nine `guidance/redesign/section-NN.md` reports.

The earlier course's research, tooling, hardware, and authorship record is preserved in [archive/2026-09-08-research-basis.md](archive/2026-09-08-research-basis.md). It is historical provenance, not an active Rustlings, authoring, or validation instruction.

## Evidence categories

Keep these categories explicit in research notes and learner-facing copy:

1. **Empirical learning evidence:** reports what participants learned under a studied intervention, with method, sample, comparison, outcome, and limits.
2. **Technical primary evidence:** establishes an algorithm, implementation, standard, measured result, or known failure mode. It does not establish pedagogy.
3. **Curriculum or project precedent:** shows that an instructional or engineering pattern is feasible. Popularity, ratings, stars, and testimonials do not establish effectiveness.
4. **Local course evidence:** comes from a named deterministic fixture, test, benchmark, hardware run, or learner artifact in this repository.
5. **Design judgment or user requirement:** records a choice made for this learner and product. It must not be phrased as an experimental result.

No study reviewed directly validates this complete adult self-study Rust ML course. Transfer strength and counterevidence must remain visible when a learning-science source comes from mathematics, physics, school classrooms, prose retrieval, or collaborative instruction.

## Source priority

Prefer, in order:

- original peer-reviewed papers and official standards;
- official documentation and original project repositories;
- author or institutional preprints when the final article is unavailable;
- high-quality systematic reviews and meta-analyses for aggregate claims;
- secondary explanations only for orientation, never as the sole support for a contested technical claim.

Verify that a URL resolves to the named source and that the source actually supports the adjacent claim. Use final versions where available. Pin mutable code or documentation versions when API behavior matters. Preserve licenses and attribution for any bundled or adapted assets.

Search snippets, unsourced summaries, generated prose, citation counts, course enrollment, stars, and marketing claims are not evidence.

## Chapter and section research record

Each section owner records in `guidance/redesign/section-NN.md` for every assigned chapter:

- the original technical sources supporting the implemented algorithms;
- official API, language, shader, file-format, or hardware documentation where relevant;
- the exact local fixture or experiment used to demonstrate the claim;
- known failure modes, assumptions, and boundary conditions;
- whether a quantitative statement is externally reported, analytically modeled, or locally measured;
- the learner-owned seam and why its assessment matches the stated outcome;
- any cross-section interface whose source or numerical convention needs verification.

Chapter `meta.json` sources contain concise learner-facing notes. Chapter `research.md` may preserve deeper claim verification and author reasoning when the section report would become unwieldy. Neither is a substitute for teaching the concept in `lesson.html`.

Use enough primary sources to support the actual claims. Do not pad a bibliography or cite a famous paper for a detail it does not contain.

## Learning-design basis

The active design uses these evidence-informed decisions, with the detailed methods and limitations in [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md):

- intact examples and substantial completion before independent work;
- conditional fading that is evaluated rather than assumed effective in programming;
- retrieval with feedback and later dependency-based reuse;
- bounded prediction followed by explicit reconciliation;
- just-in-time math with mapped concrete, visual, symbolic, and Rust representations;
- interleaving after initial practice when alternatives are genuinely confusable;
- immediate compiler/test/numerical feedback and progressive explanatory hints;
- section projects that integrate already taught material;
- assessment of implementation, diagnosis, experimental judgment, explanation, and later transfer as distinct outcomes.

Counterevidence remains part of the basis: programming fading studies include null results; rushed exploration can reduce conceptual learning and curiosity; visualizations have produced concept-specific or null transfer effects; concrete-first ordering is not universally superior; spaced-retrieval classroom effects are heterogeneous; and project-based-learning reviews have serious methodological limits.

## Curriculum precedents

Google ML Crash Course, fast.ai, the scikit-learn MOOC, MIT 6.036/6.390, CMU 10-301/601, MIT 6.S191, Berkeley Data 8, DeepLearning.AI, Dive into Deep Learning, Stanford CS336, GPU MODE Triton Puzzles, and llm.c provide structural comparisons. The primary URLs and limitations are tabulated in [LEARNING_DESIGN_RESEARCH.md](LEARNING_DESIGN_RESEARCH.md).

They support feasibility of short concept-practice cycles, working artifacts before exhaustive internals, cumulative projects, supplied setup, CPU reference oracles, and aligned exercises. Their prerequisites, languages, classroom support, libraries, and workloads differ. Do not claim that their reach proves their sequence or that their pace fits this learner.

## Technical-claim rules

- State shapes, reductions, precision, data, seeds, model size, and measurement conditions.
- Separate training objective, validation selection, final evaluation, and qualitative output.
- Ask whether every feature and preprocessing statistic is available at prediction time.
- Keep CPU/reference parity separate from optimized or GPU performance.
- Report analytical bytes/operations as models, not measured time.
- Bind speed or quality numbers to the source hardware, software, model, and dataset.
- Label toy SFT, LoRA, DPO, RL, retrieval, quantization, distributed, fairness, robustness, or MoE experiments as mechanism demonstrations at their actual scale.
- Preserve negative results and cases where an expected improvement did not occur.
- Do not infer general-purpose language ability, production safety, convergence, fairness, robustness, or deployment readiness from a tiny fixture.

Model Cards, Datasheets, and HELM are useful artifact frameworks, not teaching-effectiveness studies. Technical papers for attention, FlashAttention, quantization, LoRA, DPO, Switch Transformers, state-space models, distributed training, generation, evaluation, and deployment define mechanisms and claims; their learner-facing use still follows the evidence limits above.

## Local measurement and hardware evidence

Current v2 validation status is recorded only in [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md). Do not carry pre-v2 Radeon, MNIST, language-model, framework, server, or browser pass claims forward without rerunning the relevant v2 path.

For new local evidence, record:

- exact command and source revision;
- dataset or fixture identity and split;
- seed and configuration;
- toolchain, dependency, operating-system, and hardware details when relevant;
- warmup, samples, statistic, and dispersion for performance;
- oracle, tolerances, and boundary cases for correctness;
- unsupported or unrun paths;
- what the result does and does not establish.

A real GPU milestone requires adapter acquisition, dispatch, readback, and parity. A GPU performance claim additionally requires actual timing and device metadata. A browser illustration, analytical model, CPU fallback, shader compilation, or old hardware record cannot satisfy it.

## Research handoff

Before declaring a section research-complete, verify source URLs, reconcile source terminology with [CONSISTENCY.md](CONSISTENCY.md), map claims to actual lesson steps and local checks, and list unresolved technical or evidentiary questions. Review uses [REVIEW.md](REVIEW.md). Observed results go to [REDESIGN_VALIDATION.md](REDESIGN_VALIDATION.md); research files do not certify completion.
