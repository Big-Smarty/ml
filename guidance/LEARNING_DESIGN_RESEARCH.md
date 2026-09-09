# Learning-design research for *First Principles: Machine Learning in Rust*

**Evidence review date:** 2026-09-09
**Course scope:** 56 chapters, nine cumulative sections, adult self-study
**Target learner:** comfortable with everyday Rust; beginning machine-learning implementation

## Purpose and epistemic status

This report records the evidence used to redesign the course. It separates four different grounds for a decision:

1. **Reported evidence:** an empirical result or conclusion supported by the cited study.
2. **Contextual precedent:** a pattern used in an established curriculum or technical project, without a comparative learning study.
3. **Design judgment:** the course team's application of evidence and constraints to this particular learner and codebase.
4. **User requirement:** a settled product choice that does not need an empirical claim to justify it.

That separation matters. No controlled study located in this review directly evaluates a private, browser-supported, self-directed course that teaches the full machine-learning stack in Rust, from scalar models through GPU kernels and sparse language models. Evidence transfers with different strength from programming education, undergraduate STEM, mathematics, physics, multimedia learning, and classroom project work. The report therefore states where transfer is strong, plausible, or weak.

The detailed 56-chapter implementation matrix is being authored as a required final deliverable in the nine implementation-owner section reports under `guidance/redesign/`. Those reports specify each chapter's sessions, learner-owned algorithmic seams, local checks, transfer task, section-project contribution, and technical sources. The current ownership map is recorded in [ASSIGNMENTS.md](ASSIGNMENTS.md). This document governs the evidence and shared instructional decisions; it does not duplicate that changing implementation matrix.

## Executive conclusions

The strongest case is for a staged learning path:

> runnable example → concrete trace → just-in-time explanation and math → substantial completion → controlled variation or diagnosis → delayed reuse → section project

The evidence supports each early link more strongly than it supports the whole combined course design.

- **Worked examples** have strong aggregate evidence for helping novices with acquisition and near transfer. Far transfer is less reliable and needs explanation, varied cases, and later independent work.
- **Fading support** is a plausible bridge from examples to independent work, but programming studies are mixed: some show benefits, while Bauer and Shin report null or inconsistent fading effects. The course treats its fading pattern as a conditional design hypothesis and measures whether learners later work without the scaffold.
- **Active learning** generally outperforms lecture-only instruction in undergraduate STEM, but activity must expose thinking. Running code, moving a slider, or filling a single expression does not by itself constitute useful active learning.
- **Retrieval with feedback and distributed reuse** improves delayed retention. Transfer is conditional on response similarity, initial success, and explanatory feedback. The best spacing interval depends on the desired retention period.
- **Prediction and exploration** can prepare learners to notice an explanation, but rushed or underspecified exploration can reduce learning and curiosity. Prediction is a bounded tool, not a mandatory ritual.
- **Concrete and visual representations** can ground notation, but concrete-first order is not universally superior. Explicit correspondence among values, diagrams, notation, and Rust is the defensible design principle.
- **Interleaving** is useful when learners must distinguish confusable alternatives after initial blocked practice. Randomly shuffling unrelated topics has no comparable support.
- **Projects** can integrate knowledge and motivate revision, but the project-based-learning evidence is heterogeneous and methodologically weak. Projects should consolidate taught ideas rather than carry the burden of first instruction.
- **Readable, local assessment** should test the course's claimed capabilities: implement, debug, interpret, compare, and design a controlled experiment. Completion, confidence, training loss, and passing a demonstrated case are inadequate proxies for mastery.

The decision to keep Rust, preserve all 56 chapter topics and nine sections, use 30–45-minute sessions, and build one project per section comes from the user's goals. Research informs how those choices are implemented; it does not show that Rust or this exact section structure is superior to other languages or curricula.

## Evidence-strength framework

| Strength for this course | Meaning | Typical evidence in this review |
|---|---|---|
| **Stronger transfer** | Same broad learner behavior and a well-replicated result, though not the full course context | worked examples, retrieval with feedback, immediate task feedback, spatial integration of mutually dependent information |
| **Moderate transfer** | Related domain or mixed moderators; useful when the boundary conditions are implemented | conditional guidance fading, subgoal labels, example variability, self-explanation, interleaving for discrimination, contrasting cases, deliberate prediction |
| **Tentative transfer** | Evidence is indirect, narrow, heterogeneous, or contradicted in some settings | productive failure in Rust ML, universal concrete-first sequencing, solo browser simulations, section projects as superior pedagogy |
| **Precedent only** | An established course or project demonstrates feasibility and a useful pattern, but not causal effectiveness | fast.ai, Google ML Crash Course, scikit-learn MOOC, D2L, CS336, llm.c, GPU MODE puzzles |

Technical papers establish what an algorithm does and what claims are scientifically valid. They do not establish how the algorithm should be taught.

## Baseline course diagnosis

The pre-redesign repository audit found technically careful lessons and reference implementations, but weak alignment between stated outcomes and learner work. The old material usually behaved as a reference book followed by either a very small expression-level exercise or a large completed program. Many starters exposed only a small helper while references contained hundreds or more than a thousand lines, often dominated by CLI, parsing, allocation, serialization, device setup, or orchestration.

The primary instructional problem was therefore not a lack of information. It was the missing bridge between reading an explanation and independently building the promised system. Examples included:

- a learner computing one convolution output dimension before facing a complete CNN and backward pass;
- a one-line residual addition standing in for normalization, skip-path reasoning, and residual training;
- a scalar derivative standing in for an actual GPU training step;
- a clipping helper standing in for AdamW, scheduling, accumulation, and checkpoint-resume behavior;
- model-evaluation outcomes assessed by a single local formula rather than a split, baseline, selection, and held-out protocol.

The redesign consequently retains the technical rigor and scientific limitations while replacing the active learning path. Learners work in new cumulative labs that run before edits, modify meaningful algorithmic seams, receive readable local evidence, and progress toward section-scale independent projects. Supplied code owns incidental infrastructure; the learner owns the ML or systems mechanism named by the outcome.

## Worked examples, completion, and fading

### What studies report

Sweller and Cooper's algebra experiments compared conventional problem solving with worked-example study. Across five experiments and four instructional comparisons involving 106 Year 8–9 students, worked examples improved acquisition and speed or accuracy on same-structure transfer tasks. Structurally dissimilar transfer was not reliably improved. The study is old and narrow, but it establishes the important distinction between efficient acquisition and far transfer ([Sweller & Cooper, 1985](https://doi.org/10.1207/s1532690xci0201_3)).

Van Merriënboer's programming study compared program generation with completion and modification in ten COMAL-80 lessons. Matched voluntary groups of 28 and 29 secondary learners favored completion/modification for program construction and attrition. The sample was small, nonrandom, and used an obsolete language, but the task is directly relevant to the difference between editing a working program and generating one from nothing ([van Merriënboer, 1990](https://doi.org/10.2190/4NK5-17L7-TWQV-1EHL)).

Renkl and colleagues studied stepwise fading in a classroom quasi-experiment and two randomized laboratory studies with samples of 35, 54, and 45. Fading improved near transfer across studies, with fewer errors partly mediating the result; far-transfer findings were inconsistent ([Renkl et al., 2002](https://doi.org/10.1080/00220970209599510)). Renkl and Atkinson's review describes backward fading: omit the final solution step first, then progressively earlier steps, until the learner solves the whole problem. The reviewed evidence favored this transition for novices, while far transfer benefited from prompts that direct learners to underlying principles ([Renkl & Atkinson, 2003](https://doi.org/10.1207/S15326985EP3801_3)).

Atkinson, Renkl, and Merrill combined fading with principle-identification prompts. In a randomized first experiment with 78 learners, fading and principle prompts improved transfer; a second experiment with 40 learners replicated the prompt effect but not every fading result. The intervention was short, used one structured domain, and included no delayed test ([Atkinson, Renkl, & Merrill, 2003](https://doi.org/10.1037/0022-0663.95.4.774)).

A later worked-example meta-analysis covered 43 articles, 55 studies, and 181 effects. Its average effect was Hedges' *g*=.48 and bias-adjusted estimate *g*=.44, but heterogeneity was extremely high, *I²*=93.72%, with evidence of publication bias. The average supports worked examples in aggregate, not one optimal example design for this course ([Barbieri et al., 2023](https://doi.org/10.1007/s10648-023-09745-1)).

The expertise-reversal literature reports that supports useful to novices can become redundant and sometimes harmful to learners who already possess the relevant schema ([Kalyuga et al., 2003](https://doi.org/10.1207/S15326985EP3801_4)). A 2025 PRISMA meta-analysis of 60 randomized studies, 5,924 learners, and 176 effects similarly found that low-prior-knowledge learners benefited from more assistance while higher-prior-knowledge learners benefited from less. Assistance types and effects were heterogeneous ([Tetzlaff et al., 2025](https://doi.org/10.1016/j.learninstruc.2025.102142)). Expertise is domain-specific: fluency with borrowing and iterators does not imply fluency with reverse-mode accumulation or masked attention.

Example variability also matters. Paas and van Merriënboer found that varied examples improved transfer and learning efficiency in a geometry/CNC-programming task compared with low variability. This was one vocational study, so it supports controlled variation more than a universal prescription ([Paas & van Merriënboer, 1994](https://doi.org/10.1037/0022-0663.86.1.122)).

### Programming-specific counterevidence

Subgoal labels can make the structure of an example more visible. Margulieux and Catrambone found benefits when subgoal labels appeared in both expository text and worked examples; labels in prose supported articulation and labels in examples supported application to novel programming tasks ([Margulieux & Catrambone, 2016](https://doi.org/10.1016/j.learninstruc.2015.12.002)). A randomized App Inventor study with 40 participants reported more novel-task parts correct and faster solutions under subgoal labeling, but its sample and domain were small ([Margulieux et al., 2016](https://doi.org/10.1080/08993408.2016.1144429)).

Semester-scale findings are more restrained. A nonrandom Java CS1 study with 265 students associated subgoal materials with better formative quizzes and fewer missing assessments, but found no significant average examination advantage ([Margulieux et al., 2020](https://doi.org/10.1186/s40594-020-00222-7)). A large CS1 study by Bauer et al. found no consistent fading benefit ([Bauer et al., 2018](https://dada.cs.washington.edu/research/tr/2018/09/UW-CSE-18-09-01.pdf)), and Shin et al. found useful metacognitive-prompt effects without a significant main effect for fading ([Shin et al., 2023](https://doi.org/10.1177/07356331231174454)). These results rule out claims that fading alone guarantees programming mastery.

PRIMM—Predict, Run, Investigate, Modify, Make—offers a close structural precedent. Its reported evaluation involved 493 learners aged 11–14 across 13 schools in a teacher-mediated, quasi-experimental setting. It supports moving from reading/running code to modification and construction, but it does not validate the sequence for adult independent ML learning ([Sentance et al., 2019](https://primmportal.com/wp-content/uploads/2020/10/teaching-computer-programming-with-primm-a-sociocultural-perspective.pdf)).

### Course decision

For each unfamiliar mechanism:

1. Run an intact, meaningful baseline.
2. Trace its important values, shapes, or state transitions.
3. Label subgoals by purpose rather than syntax.
4. Ask the learner to complete a substantial final conceptual step.
5. Remove earlier support in later sessions.
6. Change one meaningful condition or diagnose a deliberately introduced fault.
7. Reuse the idea in a section project with less procedural guidance.

Fading must never collapse into a one-token hole or a single missing expression. The learner should implement a coherent algorithmic responsibility: a gradient accumulator, split policy, convolution loop, reduction kernel, cached-attention step, routing-and-capacity rule, or equivalent seam. The baseline still runs and teaches something before that work begins.

## Active learning, interaction, and visual explanation

### What studies report

Freeman et al. synthesized 225 undergraduate STEM studies. Active-learning conditions improved examination and concept-inventory performance by about 0.47 standard deviations; average failure was 33.8% under traditional lecture and 21.8% under active learning. The studies used heterogeneous activities and varied in quality, so the result establishes a broad advantage over lecture-only teaching without identifying a winning widget, project format, or ML activity ([Freeman et al., 2014](https://doi.org/10.1073/pnas.1319030111)).

Deslauriers et al. used a randomized crossover design in introductory physics. Learners in the active condition learned more but felt that they learned less, apparently interpreting effort and reduced fluency as poorer instruction. The active condition was strongly teacher-guided and included explanations; it was not minimally guided discovery ([Deslauriers et al., 2019](https://doi.org/10.1073/pnas.1821936116)).

The ICAP framework distinguishes passive reception, active manipulation, constructive production, and jointly interactive dialogue. It is a theoretical and synthesizing framework, not proof that any browser control produces learning. A solo slider is normally active manipulation at most; requiring an explanation or a new prediction makes the task more constructive ([Chi & Wylie, 2014](https://doi.org/10.1080/00461520.2014.965823)).

Animation evidence is conditional. Höffler and Leutner's meta-analysis found an average advantage for animation, strongly moderated by whether the animation represented relevant change and procedural or motor knowledge. Tversky, Morrison, and Bétrancourt warned that many animations fail to outperform well-designed static graphics. Movement helps when the movement itself carries causal information; otherwise small multiples are often clearer ([Höffler & Leutner, 2007](https://www.leibniz-ipn.de/en/research/publications/instructional-animation-versus-static-pictures-a-meta-analysis); [Tversky et al., 2002](https://doi.org/10.1006/ijhc.2002.1017)).

Simulation findings should not be generalized. In an inquiry circuits course, sections using a simulation outperformed physical-equipment sections on explanations, delayed exam items, and later circuit assembly. Section assignment and differing pre-labs weaken causal interpretation, and the result concerns one physics domain ([Finkelstein et al., 2005](https://doi.org/10.1103/PhysRevSTPER.1.010103)). A controlled 2025 ML visualization study with 40 participants improved PCA knowledge but did not improve gradient-descent knowledge or motivation. It used the same questions immediately before and after the intervention, and the static gradient-descent material omitted one of the three figures shown in the interactive condition, so it was not a pure comparison of interactivity. The result shows that benefits can be concept-specific while leaving the source of the PCA difference uncertain ([interactive ML visualization study, 2025](https://doi.org/10.1145/3724363.3729032)). An RNN visualization experiment with 37 deep-learning students reduced reported extraneous load but did not improve comprehension or transfer; text improved recall ([exploRNN study](https://doi.org/10.1007/s00371-022-02593-0)).

PhET's interview-based design research found that too many controls can overwhelm learners and motion can be watched without useful reasoning. Immediate, constrained controls and limited text supported exploration. It also documents a case where learners trusted a surprising simulation output caused by an unnoticed default-value difference. This is qualitative physics evidence, but it motivates visible held-constant values, units, assumptions, deterministic reset, and named presets ([PhET interview research](https://phet.colorado.edu/publications/archive/Phet%20Interview%20Paper.htm)).

TensorFlow Playground, Bret Victor's explorable explanations, Seeing Theory, and Distill demonstrate useful editorial patterns: reversible state, linked views, counterexamples, saved presets, explicit limitations, and a readable static path. Their popularity and design accounts are not controlled evidence of durable learning ([TensorFlow Playground paper](https://arxiv.org/abs/1708.03788); [Explorable Explanations](https://worrydream.com/ExplorableExplanations/); [Seeing Theory](https://seeing-theory.brown.edu/); [Distill on interactive articles](https://distill.pub/2020/communicating-with-interactive-articles/)).

### Course decision

A full browser lab follows this contract:

1. **Question:** name one phenomenon and one learning objective.
2. **Predict:** record an expected direction, value, or failure cause.
3. **Observe:** change one variable or choose one named comparison while held-constant values remain visible.
4. **Explain:** show the numerical difference and request one causal or transfer explanation.
5. **Build:** point to the exact local Rust file, function, command, and expected evidence.
6. **Explore:** offer and recommend broader manipulation after the guided comparison while keeping every control freely available throughout; the guided sequence creates no access gate.

No click, drag, or animation frame establishes completion by itself. The browser labels its result as an illustration using the chapter fixture; it never claims to execute or verify Rust. Simulated byte counts and conceptual GPU timelines are labeled as analytical models, not hardware measurements.

The design uses 18 purposeful interactions across the nine sections rather than one custom visualization per chapter. Other chapters use lightweight prediction, retrieval, trace, and diagnosis cards. The implementation-owner section reports contain the exact mapping. This limit reflects evidence and maintenance cost: custom interaction is justified when linked representations or counterfactual manipulation reveal an otherwise hidden mechanism.

## Prediction, contrasting cases, and productive failure

### What studies report

Schwartz and Bransford tested preparation for future learning in three college-classroom studies. Learners who analyzed contrasting cases before a lecture or text outperformed read-only or summarize-first comparisons on later learning. The cases made deep features available for the subsequent explanation, but the samples and psychology materials were limited ([Schwartz & Bransford, 1998](https://doi.org/10.1207/s1532690xci1604_4)). Later controlled classroom experiments with 128 and 120 learners found that inventing formulas from carefully engineered contrasting cases before canonical instruction matched routine practice and improved structural transfer. This was guided preparation, not free discovery ([Schwartz et al., 2011](https://doi.org/10.1037/a0025140)).

Sinha and Kapur meta-analyzed 53 studies and 166 comparisons of problem solving followed by instruction (PS-I) against instruction followed by problem solving (I-PS). Conceptual knowledge and transfer favored PS-I with unadjusted Hedges' *g*=.36, 95% CI [.20, .51]. Procedural knowledge did not differ, *g*=-.03, 95% CI [-.20, .15]. Stronger results were associated with higher-fidelity Productive Failure designs: relevant prior-knowledge activation, multiple representations or candidate solutions, attention to critical features, explanation, and consolidation that compares learner ideas with the canonical solution ([Sinha & Kapur, 2021](https://doi.org/10.3102/00346543211019105)).

The same meta-analysis gives reasons for restraint. Roughly three quarters of included comparisons concerned mathematics and physics. Evidence for professionals, learners with difficulties, and other domains was scarce. Task complexity may reduce the advantage. Many reports did not state actual failure rates, and individual differences were not available for moderator analysis.

Kapur's earlier productive-failure study used collaborative, ill-structured kinematics problems with 11th-grade learners. Subsequent conceptual and transfer performance favored the problem-first group, but group discourse itself supplied emergent guidance and the mechanism remained uncertain ([Kapur, 2008](https://doi.org/10.1080/07370000802212669)).

A 2026 randomized chemistry-simulation study provides direct counterevidence to routine exploration-first design. A 15-minute exploration-first activity reduced conceptual learning and curiosity relative to instruction first, with no basic or transfer advantage. A second experiment used a longer 20-minute exploration with stronger guidance and “why” prompts; that version improved conceptual learning, transfer, and motivation. The result shows that duration, guidance, and task design can reverse the effect ([DeCaro et al., 2026](https://doi.org/10.1111/bjep.70007)).

Prediction can focus observation. In physics lecture demonstrations, predicting first made a correct observation roughly 20–23% more likely regardless of prediction correctness. Learners who observed incorrectly often misremembered what occurred. The study concerned classroom demonstrations rather than self-directed code ([Miller et al., 2013](https://doi.org/10.1103/PhysRevSTPER.9.020113)).

### Course decision

Prediction is used when the learner has enough knowledge to interpret the result. It is small, bounded, and immediately reconciled with an actual trace or measurement. The prediction remains visible beside the observation. The lesson asks what changed, why, and what remained fixed.

Preparation-first activities use engineered contrasting cases or a short approximate solution, then supply explicit consolidation. They are used mainly for conceptual distinctions and transfer, not as a replacement for demonstrating a procedural algorithm. Long unguided guessing, blank-file exploration, and “struggle, then reveal the answer” do not meet the Productive Failure conditions and are excluded.

## Retrieval, feedback, spacing, and interleaving

### Retrieval and transfer

Roediger and Karpicke compared restudy with retrieval for college learners studying prose. Restudy performed better after five minutes; retrieval performed better after two days and one week. The result is an important warning against evaluating a learning design only by immediate fluency ([Roediger & Karpicke, 2006](https://doi.org/10.1111/j.1467-9280.2006.01693.x)).

Agarwal, Nunes, and Blunt reviewed 50 classroom experiments with 5,374 learners. Retrieval practice was beneficial across settings, and 57% of effects were medium or large. The evidence base was predominantly WEIRD, showed publication-bias concerns, and did not determine one optimal feedback schedule ([Agarwal et al., 2021](https://doi.org/10.1007/s10648-021-09595-9)). Dunlosky et al.'s broader review rated practice testing and distributed practice as high-utility techniques, while noting that implementation and task conditions matter ([Dunlosky et al., 2013](https://doi.org/10.1177/1529100612453266)).

Pan and Rickard meta-analyzed 192 transfer effects from 122 experiments. Retrieval's mean transfer advantage over re-exposure was about *d*=.40, but transfer was stronger when practice and transfer responses were congruent, initial retrieval success was high, and feedback was elaborated. Their model did not justify assuming transfer when these moderators were absent ([Pan & Rickard, 2018](https://doi.org/10.1037/bul0000151)).

### Feedback

Butler, Karpicke, and Roediger found that feedback corrected errors and reduced persistence of multiple-choice lures. In one experiment, one-day delayed feedback produced better one-week retention than immediate feedback. This result concerns prose and multiple-choice learning; it does not justify delaying compiler errors, tests, or numerical diagnostics during complex debugging ([Butler et al., 2007](https://doi.org/10.1037/1076-898X.13.4.273)).

Keuning, Jeuring, and Heeren reviewed 101 automated programming-feedback systems. Most identified an error or returned a result; fewer helped the learner repair the problem or select a useful next step. Evaluation quality and methods varied substantially ([Keuning et al., 2018](https://doi.org/10.1145/3231711)).

PeerStudio's randomized field study of 104 learners found that feedback received while revision remained live improved final scores by 4.4 percentage points; 24-hour delayed feedback resembled no feedback. The outcome was MOOC essay rubric performance, the effect was small, and peer feedback differs from code diagnostics ([Kulkarni et al., 2015](https://doi.org/10.1145/2724660.2724670)).

### Spacing

Cepeda et al. studied more than 1,350 participants learning facts and found that the best gap depends on the intended retention interval. Their temporal ridgeline is incompatible with one universal schedule such as 1/3/7 days ([Cepeda et al., 2008](https://doi.org/10.1111/j.1467-9280.2008.02209.x)).

Bego et al. aggregated a common spaced-retrieval intervention across nine introductory STEM courses. The pooled end-of-semester criterial-test gain was 2.06 percentage points, 95% CI [.16, 3.97], with high heterogeneity, *I²*=89.2%; only two courses individually showed significant effects. Excluding calculus produced an eight-course estimate of 1.50 percentage points, 95% CI [-.18, 3.26], which was not significant. It supports a low-cost pilot, not a strong universal effect ([Bego et al., 2024](https://doi.org/10.1186/s40594-024-00468-5)).

### Interleaving

Brunmair and Richter synthesized 59 studies and 238 effects. The overall interleaving benefit was *g*=.42, but mathematics showed a positive effect around *g*=.34, expository texts were nonsignificant, and word learning favored blocking. Benefits were associated with high similarity between categories and low similarity within a category, consistent with discrimination learning ([Brunmair & Richter, 2019](https://doi.org/10.1037/bul0000209)).

### Course decision

- Start a session with one to three short retrieval prompts about concepts needed now.
- Include shapes, traces, causal relationships, error diagnoses, and code decisions, not vocabulary alone.
- Give immediate compiler, test, numerical, and benchmark feedback.
- Reveal an authored explanation after an attempt, while allowing the learner to inspect it at any time; progress is never navigation-gated.
- Use a learner-controlled hint ladder: goal → invariant/shape → code location → partial structure → complete explained solution.
- Reuse a concept at its next dependency, at a mid-section comparison, and in the section review or project. This is a course-event schedule, not a claim of optimal temporal spacing.
- Introduce a method in a short blocked sequence before interleaving it with genuinely confusable alternatives.

The five-level hint ladder is a practical synthesis rather than a directly validated universal order. If a learner repeatedly needs the final levels, the right response is another worked trace or smaller subgoal, not a larger stream of raw error messages.

## Concrete, visual, symbolic, and Rust representations

### What studies report

Fyfe et al. reviewed “concreteness fading”: begin with a meaningful concrete representation, move to a more idealized or visual representation, then connect to abstract notation. Proposed benefits include grounding opaque symbols, retaining useful mental imagery, and gradually stripping incidental features. The authors explicitly describe empirical support as indirect and concentrated in mathematics and science ([Fyfe et al., 2014](https://doi.org/10.1007/s10648-014-9249-3)).

Kokkonen and Schalk's conceptual analysis argues that one representation order does not fit every science domain. What counts as concrete depends on prior knowledge and learning goals; relational mapping among representations is central ([Kokkonen & Schalk, 2021](https://doi.org/10.1007/s10648-020-09581-7)). A secondary-physics experiment found equivalent learning for concreteness fading and the reverse sequence, directly questioning a universal concrete-first advantage ([Hoogerheide et al., 2022](https://doi.org/10.1016/j.learninstruc.2021.101524)).

Spatial-contiguity research supports placing mutually dependent words and diagrams together so learners do not spend capacity searching and matching across sources. Redundant information can itself become costly, especially for more knowledgeable learners ([Ginns, 2006](https://doi.org/10.1016/j.learninstruc.2006.10.001)).

### Course decision

Just-in-time mathematics normally moves through:

1. actual values, arrays, shapes, tokens, or a tiny running case;
2. a trace table, annotated diagram, or plotted relationship;
3. formal notation with every symbol and reduction defined;
4. the corresponding Rust expression, loop, or shader operation;
5. a return from the implementation and output to the notation.

The page explicitly maps axis to index, tensor dimension to runtime shape, formula term to expression, gradient factor to update, or byte count to buffer. It eventually removes the concrete case so the learner must use the abstract relation on a new input.

This is a design judgment built on representational mapping evidence. The report does not claim that concrete-first order is always best. Concrete examples can distract through surface features, and multiple representations can increase load when their relationship is unclear.

## Self-explanation and experimental judgment

Chi, de Leeuw, Chiu, and LaVancher compared 14 prompted eighth-grade biology learners with 10 controls. Prompted learners developed better mental models and gains, but study time was roughly doubled and the sample was very small ([Chi et al., 1994](https://doi.org/10.1207/s15516709cog1803_3)). Self-explanation is therefore concentrated at pivotal subgoals: an invariant, surprising output, tradeoff, or reason an alternative fails. The course does not demand paraphrase after every line.

Holmes, Wieman, and Bonn studied successive physics-lab cohorts of roughly 130 learners per condition. Repeated prompts to compare data with a model, decide, and revise led to 12 times more spontaneous method improvement and roughly four times more identification and explanation of limitations after prompts were removed; behavior persisted later in the course. The prior-year comparison was not a concurrent randomized trial, and the tasks were physics experiments ([Holmes et al., 2015](https://doi.org/10.1073/pnas.1505329112)).

The course applies that experimental-judgment sequence to model and systems work:

> hypothesis → controlled change → recorded evidence → discrepancy → revision → stated limitation

Every section project records the seed and configuration, baseline, intervention, invariant, metric, observed result, interpretation, and limitation. This turns experimentation into an assessed capability rather than a decorative chart.

## Projects and cumulative structure

### What studies report

Barron et al. derived project-design principles from school-based work: learning-appropriate goals, scaffolds and contrasting cases, formative self-assessment with revision, and social or agency structures. Their principal setting was fifth-grade collaborative classroom work, so the paper supports structured projects more than independent adult web projects ([Barron et al., 1998](https://doi.org/10.1080/10508406.1998.9672056)).

A 2026 umbrella review covered 15 meta-analyses and 351 unique studies. Results generally favored project-based learning, but reported magnitudes ranged widely, project-based and problem-based designs were often conflated, and all 15 meta-analyses were rated critically low under AMSTAR 2. Directional optimism is reasonable; a strong general causal claim is not ([Farshad & Fortin, 2026](https://doi.org/10.1016/j.edurev.2026.100809)).

Raschka describes a senior-undergraduate deep-learning course using full demonstrations, fill-in skeletons, project reuse, proposals, reports, presentations, and peer review. Course ratings and comments were positive, but the author reports no A/B test or formal causal evaluation ([Raschka, 2022](https://proceedings.mlr.press/v170/raschka22a/raschka22a.pdf)).

Small interface-design experiments suggest benefits from producing and critiquing parallel alternatives before commitment, but their domains do not establish ML learning. Dow et al. studied 33 individuals and later 84 pairs; Tohidi et al. studied 48 participants ([Dow et al., 2010](https://doi.org/10.1145/1879831.1879836); [Dow et al., 2011](https://doi.org/10.1145/1978942.1979359); [Tohidi et al., 2006](https://doi.org/10.1145/1124772.1124960)). The safe application is to compare several cheap model or configuration candidates before making an expensive commitment.

### Course decision

The nine section projects are cumulative integration environments. They are not used as first exposure to a mechanism. Each project provides:

- a meaningful objective using already introduced concepts;
- a runnable baseline with a visible limitation;
- parsing, data access, CLI, serialization, visualization, allocation, host/device, or server plumbing where it is incidental;
- stable interfaces around learner-owned ML or systems algorithms;
- readable local checks and at least one unfamiliar but inspectable transfer case;
- revision after feedback;
- a small required core and clearly optional extensions;
- a reference implementation available for comparison after an attempt;
- a stop condition based on evidence rather than time spent.

Later projects fade implementation recipes and recommended choices while retaining constraints, interfaces, and validation. Independent work means selecting and integrating taught concepts. It does not require reconstructing incidental infrastructure.

### Nine cumulative project arcs

The project themes are a curriculum design judgment. They preserve the existing section structure while giving repeated concepts a shared dataset, interface, or validation protocol within each section.

| Section | Cumulative project role | How responsibility fades |
|---|---|---|
| 01–06, First principles | Fit and evaluate a noisy sensor model, progress from one input to classification and reliable optimization | Full traces and supplied loops become learner-authored prediction, gradient, split, metric, and experiment decisions |
| 07–11, Build a neural network | Move from XOR failure through backpropagation, autodiff, tensors, digit classification, optimizers, and checkpoints | Manual worked backward traces become reusable learner-authored mechanisms inside supplied data and checkpoint plumbing |
| 12–19, Beyond neural networks | Analyze one maintenance-style table with uncertainty, grouped/time-aware splits, classical models, PCA, clustering, and fair selection | Guided algorithm comparisons become an independent, budgeted model-selection and error-analysis report |
| 20–24, Broader deep learning | Compare representations for vision, sequence, and recommendation problems under a common experiment/report contract | Supplied forward models become learner-owned core operations and a domain-chosen independent experiment |
| 25–28, Make the CPU faster | Turn one correct dense/GEMM kernel into an honestly measured local fast path | Correct scalar oracle and benchmark harness remain; learner owns locality, partition, reduction, SIMD/tail, and interpretation |
| 29–32, Write GPU kernels | Carry one supplied wgpu pipeline through dispatch, reduction/GEMM, training, fusion, and measurement | Host/device plumbing and CPU oracles remain; shader logic and hardware evidence become increasingly independent |
| 33–39, Train your language model | Build a byte-level model through tokenizer comparison, attention, decoder, data, training, and resume | Known tiny checkpoints become a learner-designed fixed-budget language-model experiment; BPE remains a substantive comparison rather than a forced capstone integration |
| 40–46, Efficient and adapted LLMs | Compare caching, quantization, efficient attention, adapters, retrieval, toy policy learning, and preference objectives under controlled fixtures | Early mechanisms use full parity traces; later work asks the learner to choose metrics and interpret quality/resource limits |
| 47–56, Advanced architectures and engineering | Compare sparse, expert-routed, recurrent, distributed, generative, multimodal, production, evaluation, and portability mechanisms, ending in a sparse MoE LM | Stable dense references and infrastructure remain while design, integration, failure analysis, and final evidence become learner-owned |

The chapter-level sessions and exact check contracts are being completed in the implementation-owner reports under `guidance/redesign/`; they remain a required final deliverable. Current ownership is available in [ASSIGNMENTS.md](ASSIGNMENTS.md). Keeping the matrices in the section reports allows technical authors to update interfaces and numerical fixtures without weakening or rewriting the evidence claims in this document.

## Comparison with established curricula

These curricula show that the proposed patterns are feasible and recognizable. Their public pages generally provide structure, prerequisites, exercises, and artifacts rather than controlled comparisons. Enrollment, ratings, badges, stars, and testimonials are reach or satisfaction indicators, not learning evidence.

| Curriculum | Observed pattern | Useful precedent | Why direct adoption would be invalid |
|---|---|---|---|
| [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) | Short explanations, checks, browser visuals, Colab exercises, quizzes; regression and classification before broader topics | Small concept–practice cycles, direct manipulation, early evaluation discipline | Requires programming, Python/NumPy/pandas, algebra and statistics; no comparative evidence that its sequence produces durable transfer |
| [fast.ai Practical Deep Learning for Coders](https://course.fast.ai/) | Useful end-to-end model early, foundations and from-scratch work later, executable notebooks and projects | Working-result-first motivation and just-in-time mathematical explanation | Assumes coding experience and high-level Python libraries; testimonials and artifacts do not establish causal effectiveness for Rust foundations |
| [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | Predictive pipeline, validation, tuning, linear models, trees, ensembles; explanation → exercise → solution → quiz → takeaways | Cumulative tabular pipeline, train-only preprocessing, repeated feedback rhythm | Assumes Python and scientific libraries; not a comparison against other instructional designs |
| [MIT 6.036/6.390](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/) | Notes and preparation, staff-supported labs, substantial homework and examinations; from-scratch models | Alignment of explanation, practice, lab, and independent assessment | Prerequisites include programming, matrices and gradients; classroom support and workload differ sharply from self-study |
| [CMU 10-301/601](https://www.cs.cmu.edu/~hchai2/courses/10601/) | Core algorithms, empirical questions, quizzes, recitations, substantial assignments | Learner-authored algorithms plus experimental interpretation | Expects programming volume, probability/statistics and discrete mathematics; pace is not a beginner model |
| [MIT 6.S191](https://introtodeeplearning.com/) | Intensive lectures and software labs ending in a project | Visible project outcomes and aligned implementation work | Assumes mathematical and software fluency; bootcamp pace cannot justify this course's pacing |
| [Berkeley Data 8](https://data8.org/) | Authentic questions, constrained API, hosted notebooks, labs, homework, projects, inline feedback | Low-setup data work and deliberately supplied infrastructure | It is a statistics/data-science foundation, not a full ML implementation curriculum; evaluations do not isolate the proposed design |
| [DeepLearning.AI Machine Learning Specialization](https://www.deeplearning.ai/specializations/machine-learning) | Intuition and visualization, implementation, then optional deeper mathematics | Layering intuition, code, and math | Uses Python and platform infrastructure; popularity and ratings do not establish transfer to Rust systems work |
| [Dive into Deep Learning](https://d2l.ai/) | Exposition, figures, mathematics, runnable code; broad topics split into smaller conceptual units | Keep deep topics while dividing them into several sessions | Library-centered multi-language book with different support and prerequisites; no causal sequence comparison |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/) | Tokenizer/model/optimizer, profiling/kernels/distribution, scaling, data, alignment | Whole-system arc and CPU-first correctness before expensive GPU work | Assumes prior deep learning, mathematics, PyTorch, and systems optimization; it is a destination model rather than a novice pace model |
| [GPU MODE Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | Small indexing and masking puzzles with interpreter support | Focused tile/workgroup representations before hardware execution | Triton interpreter and puzzle progression do not substitute for actual Rust/wgpu dispatch or prove learning outcomes |
| [Karpathy's llm.c](https://github.com/karpathy/llm.c) | CPU fp32 reference beside progressively faster CUDA paths, known activations and gradients | CPU oracles and intermediate-state parity for learner kernels | Expert technical project, not a novice curriculum or comparative learning study |

The curriculum comparison supports the course's architecture by triangulation: working artifacts before exhaustive internals, recurring exercise/solution/check cycles, cumulative projects, early evaluation discipline, supplied setup, and CPU references. It does not establish that the combined redesign is “evidence proven.”

## Shared 30–45-minute session design

Thirty to forty-five minutes is a user-selected session target, not a research-derived optimal duration. A chapter may contain several sessions. The full chapter remains available as one continuous semantic document; guided mode surfaces the same sections as short steps with stable anchors.

A typical session contains:

1. **Re-enter, 2–4 minutes.** Retrieve one or two prerequisites needed now and self-check against explanatory feedback.
2. **Run and inspect, 4–7 minutes.** Execute a deterministic working baseline and identify its input, output, purpose, and visible limitation.
3. **Trace and explain, 6–10 minutes.** Work through one concrete numerical or state trace with collocated diagram, notation, and Rust.
4. **Build, 12–20 minutes.** Implement one substantial algorithmic seam in supplied plumbing, with progressive hints and immediate local feedback.
5. **Vary or diagnose, 5–8 minutes.** Change one meaningful condition or locate a deliberately introduced failure.
6. **Stop and record, 2–4 minutes.** Pass a named core check and state one inference or limitation.

This is a compositional template rather than a mandatory timer. A session may emphasize a worked trace, a build, or an independent comparison. Authors split large chapters instead of compressing or quietly dropping assigned topics.

Valid stop conditions are observable:

- “The analytic gradient agrees with central differences over the stated cases, and the learner explains why both use the same old parameter state.”
- “The GPU kernel returns CPU-parity values for odd lengths on a real adapter; without an adapter, the conceptual workgroup trace is complete but the hardware milestone remains unclaimed.”
- “The model-selection report identifies its split, baseline, search budget, chosen validation criterion, one-time final evaluation, and a representative failure.”

Reading every optional panel, minimizing loss, or spending 45 minutes is not a stop condition.

## Division of learner work and supplied plumbing

Keeping Rust is a product decision. Rust makes memory, representation, concurrency, and device boundaries explicit, which is valuable in later systems sections. The same features can consume attention before the ML concept is understood. The course therefore uses this division:

**Learner owns:**

- prediction, loss, gradient, update, tensor indexing, algorithm kernel, split or evaluation policy, model comparison, shader core, optimizer state rule, routing/capacity policy, and experimental interpretation when those are the chapter outcome;
- 2–5 coherent named functions or comparable seams rather than hundreds of incidental lines;
- diagnosis of an invariant violation and an independent controlled variation.

**Course supplies when incidental:**

- CLI and argument parsing;
- input formats, downloads, deterministic fixtures, and serialization;
- plotting and browser display;
- tensor/buffer allocation and general error formatting;
- wgpu adapter/device/pipeline/bind-group/command/readback setup unless the chapter explicitly teaches that boundary;
- HTTP, bounded file access, thread startup, channels, and checkpoint bytes;
- deterministic seeds and CPU reference oracles.

Supplying infrastructure does not reduce rigor when the check invokes the learner's real public entry point and the learner must interpret end-to-end behavior. Requiring unrelated plumbing can weaken construct validity by making a test measure Rust API recall rather than the intended ML idea.

## Assessment validity

### Intended claims

The course may reasonably collect evidence for five kinds of capability:

1. **Mechanistic understanding:** trace values, shapes, indices, gradients, or state and explain a causal relationship.
2. **Implementation:** complete a coherent algorithmic seam that works across several inputs and boundaries.
3. **Diagnosis:** identify an invariant violation from evidence and repair it.
4. **Experimental judgment:** choose a comparison, hold relevant factors fixed, use an appropriate metric, interpret uncertainty, and state limits.
5. **Integration and transfer:** reuse concepts with less guidance in a new case or section project.

The assessment system should not claim general ML expertise, production readiness, or durable mastery from completing the private course. Those broader claims would require independent and delayed assessment beyond the built-in formative environment.

### Evidence chain

Each outcome is aligned to at least one observable action and one artifact:

| Claim | Immediate evidence | Stronger follow-up evidence |
|---|---|---|
| Understands a mechanism | correct trace plus causal explanation | applies invariant to a new shape, seed, or failure |
| Can implement it | readable local checks call learner code across multiple cases | later project reuses it without the chapter recipe |
| Can debug it | diagnosis names violated invariant before edit | controlled fault with different surface symptoms |
| Can evaluate it | report includes split, baseline, metric and uncertainty | frozen held-out check and representative error analysis |
| Can reason about systems | parity, bytes/work model, environment metadata | real hardware timing/dispatch evidence where claimed |

Passing a baseline is prerequisite evidence, not mastery. Passing only a near-identical demonstrated case is near transfer. The design adds an unfamiliar but inspectable case and later reuse because worked-example research repeatedly shows that far transfer is less dependable.

### Transparent local checks

All milestone checks are local and readable. They call the learner's implementation, cover several values, shapes, seeds, or boundaries where appropriate, and print the evidence used. They avoid source-pattern matching, brittle exact optimizer trajectories, and hidden one-expression graders.

Readable tests cannot prevent a learner from inspecting the evaluation case. This course is formative, so transparency and debuggability take priority over secrecy. A final split can be protected by an explicit “evaluate once after selection” protocol, but that is an experimental norm enforced by the learner, not a secure examination. If a future credential needs a defensible proficiency claim, it should use independent unseen tasks administered outside this repository.

### Reliability and numerical evidence

- Deterministic fixtures and seeds make conceptual failures reproducible.
- Stochastic claims use several seeds or a distribution, not one favorable run.
- Floating-point checks use justified tolerances and test invariants rather than exact trajectories.
- Performance claims record hardware, toolchain, configuration, input sizes, warmup, samples, and dispersion.
- GPU capability, correctness, and performance are separate milestones. A browser model or CPU fallback cannot award a hardware-performance claim.
- Architecture-specific SIMD is evidenced only on a supported path; all learners still complete tail handling and dispatch reasoning.
- Model quality is compared with a baseline and held-out evidence. Low training loss alone is not evidence of useful generalization.

### Reasoning and explanation

Free-text explanations are self-checked against a worked answer or rubric; the site does not pretend to understand arbitrary prose. Two-tier prompts—answer plus reason—are useful because a correct choice can coexist with incorrect reasoning. Hirsch and O'Donnell found persistent representativeness misconceptions in a sample of 263 statistics learners, motivating assessment of rationale as well as answer, though their study was not a trial of this site's remediation ([Hirsch & O'Donnell, 2001](https://doi.org/10.1080/10691898.2001.11910655)). Konold's interviews with 16 undergraduates similarly documented deterministic yes/no interpretations of probability, a narrow qualitative result that motivates repeated-sampling traces rather than proving their effectiveness ([Konold, 1989](https://doi.org/10.1207/s1532690xci0601_3)).

### Progress labels

Browser progress remains literal and self-reported:

- guide viewed or completed;
- prediction recorded;
- answer checked;
- local Cargo check reported;
- retrieval self-check date.

These states remain separate. The site does not collapse them into a mastery percentage, gate navigation, infer correctness, or award progress for a control movement. Clearing browser storage loses the record and is described plainly.

This separation is supported by the discrepancy between felt and measured learning reported by Deslauriers et al. It also prevents course completion from being mistaken for independent transfer.

### Validity threats and mitigations

| Threat | Consequence | Mitigation and remaining limit |
|---|---|---|
| Example copying | passes a demonstrated case without abstraction | unfamiliar inspectable variation and delayed reuse; does not prove far transfer |
| Test inspection | learner tunes directly to cases | several property/boundary cases and honor-system final split; not secure certification |
| Infrastructure burden | measures Rust/API knowledge instead of ML | supply incidental plumbing behind stable interfaces |
| Stochastic variance | one seed creates false success/failure | deterministic teaching fixture plus multi-seed experimental evidence |
| Numerical brittleness | correct algorithms fail exact comparisons | scale-aware tolerances, invariants, and reference oracles |
| Leakage or repeated test use | inflated generalization estimate | train-only preprocessing, frozen selection protocol, one-time final evaluation |
| Training loss as proxy | memorization mistaken for learning | baselines, held-out data, label-shuffle/fault cases, error analysis |
| Browser model mistaken for execution | simulated evidence treated as Rust/hardware result | explicit illustration labels and separate local/hardware checks |
| Self-report inflation | confidence mistaken for competence | literal progress states plus later unaided tasks |
| Over-scaffolding | expert learner follows redundant steps | diagnostics, collapsible explanation, open navigation, faded project guidance |

### Leakage and evaluation literacy

Leakage can arise through future information, duplicated or grouped examples, preprocessing before the split, and model selection on evaluation data. Kaufman et al. provide a technical leakage taxonomy rather than an instructional trial ([Kaufman et al., 2012](https://doi.org/10.1145/2382577.2382579)). The scikit-learn common-pitfalls example shows random labels with 10,000 random features attaining deceptively high performance when feature selection precedes splitting, then returning to chance under the correct order ([scikit-learn leakage example](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)). Google MLCC similarly distinguishes validation iteration from final test use ([Google data splitting guidance](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)). These are technical demonstrations and official guidance, not evidence that one lesson cures leakage errors.

The course therefore treats data availability, grouping, time, preprocessing state, model selection, and one-time final evaluation as executable parts of the project contract. A static-analysis study of data leakage can establish that leakage patterns occur in real ML code, but not that the proposed exercise eliminates them ([Yang et al., 2022](https://www.cs.cmu.edu/~ckaestne/pdf/ase22.pdf)).

### Responsible-AI and documentation assessments

Model Cards, Datasheets for Datasets, and HELM provide useful technical artifact structures for recording intended use, data provenance, subgroup results, scenarios, and limitations. They are not learning-science evidence ([Model Cards](https://doi.org/10.1145/3287560.3287596); [Datasheets for Datasets](https://doi.org/10.1145/3458723); [HELM](https://arxiv.org/abs/2211.09110)). The course uses their documentation patterns as assessable artifacts, while keeping technical claims tied to actual model/data checks.

## Scientific-claim discipline for systems and advanced chapters

Later chapters create special validity risks because a tiny local demonstration cannot support claims made by large-scale research systems.

- A workgroup animation establishes indexing logic, not GPU execution.
- CPU parity establishes a kernel's numerical target, not GPU performance.
- Analytical byte or operation counts are models, not latency measurements.
- Quantized storage size does not establish speed.
- Online or tiled attention parity does not establish FlashAttention performance.
- A toy LoRA or DPO experiment demonstrates mechanics, not the full quality gains of a production alignment pipeline.
- One reinforcement-learning seed establishes mechanics, not convergence.
- Active MoE parameters do not equal FLOPs, latency, memory traffic, or quality.
- A drift statistic is a signal, not proof of harmful model behavior.
- Attribution, robustness, membership, fairness, and causal checks answer different questions and must not be collapsed into one responsibility score.

Implementation-owner reports bind each advanced claim to canonical technical sources and local evidence. This research report governs the educational claim: learners should distinguish implementation parity, modeled resource counts, measured execution, and external research results.

## Accessibility and resilience as learning conditions

Accessibility requirements are standards-based product obligations and also reduce avoidable barriers to the learning task.

- Use native controls before custom widgets.
- Every range has an exact keyboard/non-drag alternative.
- Meaning never depends on color alone.
- Dynamic charts have an adjacent text or table equivalent.
- Status messages occur after meaningful actions rather than every animation frame.
- No autoplay; temporal processes provide step, pause, and reset.
- Reduced-motion mode uses static small multiples or manual stepping.
- The continuous chapter, worked answers, commands, and code links remain usable without JavaScript.
- Guided and continuous views use the same semantic content and stable anchors.
- At 320 CSS pixels and 400% zoom, controls stack and no function is lost.

Primary references: [WCAG 2.2 keyboard](https://www.w3.org/WAI/WCAG22/Understanding/keyboard.html), [use of color](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html), [contrast minimum](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html), [reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html), [status messages](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html), [target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html), and [complex-image text alternatives](https://www.w3.org/WAI/tutorials/images/complex/).

Accessibility conformance does not prove learning effectiveness, but inaccessible operation invalidates any learning claim for excluded users.

## Decisions adopted for implementation

1. **Keep all existing topics and the nine-section sequence.** Broad chapters become several sessions rather than losing material.
2. **Use a runnable-baseline-first path.** Default `cargo run` or `cargo test` succeeds and shows a meaningful artifact before learner edits.
3. **Retire missing-expression practice as the primary path.** Learners change coherent algorithmic seams in new cumulative labs.
4. **Supply incidental plumbing.** Rust remains visible where representation, safety, memory, concurrency, or device boundaries are the topic.
5. **Use section projects for integration.** Projects follow instruction and faded practice.
6. **Use just-in-time math with explicit representational mapping.** Concrete-first is a default editorial sequence, not a universal claim.
7. **Use bounded prediction and contrasting cases.** Open exploration is optional and follows a guided pass.
8. **Keep feedback immediate and explanatory.** Authored answers and hints stay accessible; navigation is never locked.
9. **Use dependency-based retrieval and discrimination-based interleaving.** Do not claim an optimal calendar schedule.
10. **Assess implementation, diagnosis, experiment design, explanation, and later reuse separately.** Progress remains literal and self-reported.
11. **Separate conceptual, CPU-oracle, real-GPU, and real-performance evidence.** Hardware claims require hardware execution and metadata.
12. **Preserve readable static content and local execution.** Browser interactions enhance the explanation rather than becoming the sole learning path.

## Claims the course may and may not make

### Defensible claims

- Worked examples and substantial completion tasks often help novices on acquisition and near-transfer tasks; support should fade as relevant knowledge grows.
- Retrieval with feedback usually improves delayed retention, while transfer depends on how practice and later use align.
- Active work generally outperforms lecture-only instruction in undergraduate STEM, but the kind and quality of activity matter.
- A useful course interaction asks learners to predict, inspect, explain, and connect the result to executable code.
- Leakage, overfitting, numerical instability, and systems-measurement errors require explicit counterexamples and checks.
- The course design is evidence-informed and deliberately tests its stated outcomes.

### Claims to avoid

- The complete 56-chapter redesign is proven superior.
- Rust is empirically the best language for learning ML.
- Worked examples or projects guarantee far transfer.
- Productive failure is superior to instruction first in every setting.
- Concrete-first presentation is universally better.
- A visualization, simulation, animation, or slider is inherently active learning.
- A fixed retrieval interval is optimal for every concept.
- Passing readable local tests certifies independent mastery.
- Lower training loss proves generalization.
- CPU fallback or browser simulation proves GPU correctness or speed.
- A famous curriculum's popularity establishes its effectiveness.

## Primary and high-value source inventory

### Learning and instruction

- Sweller, J., & Cooper, G. A. (1985). Worked examples in algebra. https://doi.org/10.1207/s1532690xci0201_3
- van Merriënboer, J. J. G. (1990). Program completion and modification. https://doi.org/10.2190/4NK5-17L7-TWQV-1EHL
- Renkl, A., et al. (2002). Stepwise fading. https://doi.org/10.1080/00220970209599510
- Renkl, A., & Atkinson, R. K. (2003). Transition from example study to problem solving. https://doi.org/10.1207/S15326985EP3801_3
- Atkinson, R. K., Renkl, A., & Merrill, M. M. (2003). Fading and principle prompts. https://doi.org/10.1037/0022-0663.95.4.774
- Barbieri, C. A., et al. (2023). Worked-example meta-analysis. https://doi.org/10.1007/s10648-023-09745-1
- Kalyuga, S., et al. (2003). Expertise reversal. https://doi.org/10.1207/S15326985EP3801_4
- Tetzlaff, L., et al. (2025). Prior knowledge and instructional assistance meta-analysis. https://doi.org/10.1016/j.learninstruc.2025.102142
- Margulieux, L. E., & Catrambone, R. (2016). Subgoal labels in programming examples. https://doi.org/10.1016/j.learninstruc.2015.12.002
- Margulieux, L. E., et al. (2020). Subgoal examples in Java CS1. https://doi.org/10.1186/s40594-020-00222-7
- Freeman, S., et al. (2014). Active learning in undergraduate STEM. https://doi.org/10.1073/pnas.1319030111
- Deslauriers, L., et al. (2019). Measured versus felt learning. https://doi.org/10.1073/pnas.1821936116
- Chi, M. T. H., & Wylie, R. (2014). ICAP. https://doi.org/10.1080/00461520.2014.965823
- Roediger, H. L., & Karpicke, J. D. (2006). Retrieval and delayed retention. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Agarwal, P. K., et al. (2021). Classroom retrieval systematic review. https://doi.org/10.1007/s10648-021-09595-9
- Pan, S. C., & Rickard, T. C. (2018). Transfer from test-enhanced learning. https://doi.org/10.1037/bul0000151
- Cepeda, N. J., et al. (2008). Spacing and retention interval. https://doi.org/10.1111/j.1467-9280.2008.02209.x
- Brunmair, M., & Richter, T. (2019). Interleaving moderators. https://doi.org/10.1037/bul0000209
- Schwartz, D. L., & Bransford, J. D. (1998). Contrasting cases before telling. https://doi.org/10.1207/s1532690xci1604_4
- Sinha, T., & Kapur, M. (2021). Problem solving followed by instruction meta-analysis. https://doi.org/10.3102/00346543211019105
- DeCaro, M. S., et al. (2026). *Flipping a simulation before instruction can improve students’ learning, interest and perceived competence.* https://doi.org/10.1111/bjep.70007
- Fyfe, E. R., et al. (2014). Concreteness fading review. https://doi.org/10.1007/s10648-014-9249-3
- Hoogerheide, V., et al. (2022). Concreteness-order equivalence in physics. https://doi.org/10.1016/j.learninstruc.2021.101524
- Chi, M. T. H., et al. (1994). Self-explanation. https://doi.org/10.1207/s15516709cog1803_3
- Keuning, H., et al. (2018). Automated programming-feedback review. https://doi.org/10.1145/3231711
- Holmes, N. G., et al. (2015). Teaching experimental judgment. https://doi.org/10.1073/pnas.1505329112
- Barron, B. J. S., et al. (1998). Structured project learning. https://doi.org/10.1080/10508406.1998.9672056
- Farshad, N., & Fortin, C. (2026). Project-based-learning umbrella review. https://doi.org/10.1016/j.edurev.2026.100809

### Programming and interactive-learning context

- Sentance, S., et al. (2019). PRIMM classroom evaluation. https://primmportal.com/wp-content/uploads/2020/10/teaching-computer-programming-with-primm-a-sociocultural-perspective.pdf
- Margulieux, L. E., et al. (2016). Subgoal-label randomized programming study. https://doi.org/10.1080/08993408.2016.1144429
- Bauer, A., et al. (2018). CS1 fading counterevidence. https://dada.cs.washington.edu/research/tr/2018/09/UW-CSE-18-09-01.pdf
- Höffler, T. N., & Leutner, D. (2007). Animation versus static pictures meta-analysis. https://www.leibniz-ipn.de/en/research/publications/instructional-animation-versus-static-pictures-a-meta-analysis
- Tversky, B., et al. (2002). Limits of animation. https://doi.org/10.1006/ijhc.2002.1017
- Finkelstein, N. D., et al. (2005). Physics simulation comparison. https://doi.org/10.1103/PhysRevSTPER.1.010103
- Smilkov, D., et al. (2017). TensorFlow Playground design account. https://arxiv.org/abs/1708.03788

### Assessment and ML-practice context

- Kaufman, S., et al. (2012). Leakage in data mining. https://doi.org/10.1145/2382577.2382579
- Yang, J., et al. (2022). Data leakage in ML code. https://www.cs.cmu.edu/~ckaestne/pdf/ase22.pdf
- Valdenegro-Toro, M., & Sabatelli, M. (2022). Descriptive evidence of overfitting misconceptions. https://arxiv.org/abs/2209.03032
- Zhang, C., et al. (2017). Deep networks fitting random labels. https://openreview.net/forum?id=Sy8gdB9xx
- Hirsch, L. S., & O'Donnell, A. M. (2001). Answer-plus-reason probability assessment. https://doi.org/10.1080/10691898.2001.11910655
- Konold, C. (1989). Outcome-oriented probability reasoning. https://doi.org/10.1207/s1532690xci0601_3
- Mitchell, M., et al. (2019). Model Cards. https://doi.org/10.1145/3287560.3287596
- Gebru, T., et al. (2021). Datasheets for Datasets. https://doi.org/10.1145/3458723
- Liang, P., et al. (2022). HELM. https://arxiv.org/abs/2211.09110

## Final interpretation

The redesign is best understood as a disciplined application of converging evidence rather than a validated package. It gives novices a working model of the target behavior, reduces incidental implementation burden, makes the mechanism observable, transfers responsibility in substantial steps, and later asks for independent integration. It also builds in the tests needed to discover where that transfer fails.

The strongest evaluation of the redesign will come from learner evidence gathered across time: immediate implementation and explanation, later unaided reuse, section-project decisions, hint dependence, representative failures, and the difference between confidence and performance. Those observations should be used to revise the course without retroactively portraying the original design judgment as settled science.
