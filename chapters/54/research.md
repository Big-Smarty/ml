## Active redesign route (Section 09)

The active learner route is `just lab 54` → edit `labs/s09-advanced/src/ch54.rs` → `just lab-check 54`. The lesson now contains 3 sessions and 9 ordered question-led steps. Goal: Implement subgroup audit metrics and explain tested attribution, perturbation, membership, parser, and causal limits. The separately explained solution is `labs/s09-advanced/src/solutions/ch54.rs`; supplied validation, fixtures, and I/O are in `src/common/ch54.rs`. Original `projects/ch54` implementations remain preserved reference evidence. Earlier workflow descriptions below are historical reference notes and do not replace this active route.

Every session includes a concrete worked calculation, implementation checkpoint, graduated hints, an explained answer, and a changed-input transfer task. The browser illustration in chapter 48 executes only a fixed local arithmetic fixture; no browser control claims to run Rust. Executable goal checks and source-review limits are recorded in `guidance/redesign/section-09.md`.

# Chapter 54 research

Route: GPT-5.6 Luna High, bounded primary-source research pass.

Course metadata: **Responsible evaluation**. Topics are interpretability, robustness, privacy, security, fairness, and causal limits. The project audits a model and its evidence using course-authored grouped evaluation data. The notes below are author-facing evidence and scope constraints, not lecture prose.

## Claim-to-source notes

### Interpretability and feature attribution

- Adebayo et al. [1] propose model- and data-randomization sanity checks for saliency maps. Their experiments find that some popular saliency methods can remain effectively unchanged when model parameters or labels/data-generating information are randomized. The chapter may therefore teach: a visually plausible attribution is not evidence that the explanation reflects the trained model or learned data relationship; test sensitivity to the model and data before using an attribution for debugging or causal language.
- Keep the claim bounded. The paper studies saliency methods, mainly on image classifiers; it does not establish that every explanation method fails, nor that a failed randomization test alone identifies the best replacement. The audit should report the explanation method, target output, perturbation/randomization test, and what the explanation is being used to support.

### Adversarial robustness

- Madry et al. [2] frame adversarial robustness as robust optimization. A near-indistinguishable input can be deliberately changed to induce an incorrect prediction, while adversarial training can improve resistance to a specified class of attacks. The chapter may distinguish ordinary held-out accuracy from robustness under an explicit threat model, perturbation set, attacker access, and budget.
- The source’s security guarantee is tied to a defined first-order adversary and empirical evaluation. It does not justify the blanket claim that a model is “robust” against every attack, distribution shift, physical transformation, or adaptive attacker. The project should preserve the threat-model fields beside each robustness score.

### Differential privacy: definition, accounting, and noise

- Dwork and Roth [3] define differential privacy as a property of a randomized data-access mechanism over neighboring datasets, parameterized by ε and, for approximate DP, δ. They present calibrated mechanisms such as the Laplace mechanism and composition theorems. The useful teaching distinction is: noise is an implementation ingredient; DP is a quantified guarantee proved from adjacency, sensitivity, randomness, and the complete sequence of releases.
- Therefore “we added noise” is not enough to claim privacy. Arbitrary noise with unbounded or unmeasured sensitivity, an unreported privacy parameter, or repeated training/evaluation releases without composition accounting has no stated ε,δ guarantee. The audit should record the neighboring-unit definition (row, person, or group), mechanism, sensitivity/clipping assumptions, per-step/release costs, composition/accountant, and final budget. Also distinguish privacy of an output mechanism from confidentiality of the raw training process or infrastructure.
- The chapter should avoid presenting DP as a promise of zero disclosure or as a guarantee that a model cannot memorize anything. The source describes a formal stability guarantee with an accuracy/privacy trade-off and limits under many queries.

### Security and membership inference

- Shokri et al. [4] construct a black-box membership-inference attack: an attack model learns differences between a target model’s behavior on records seen during training and records not seen during training. Their experiments report vulnerability in several commercial machine-learning services and include a sensitive hospital-discharge membership setting.
- The chapter may use membership inference as a concrete security audit: define the candidate record population, attacker access, score/output interface, attack training data, and report attack advantage or an operating-point table. Do not collapse this into attribute reconstruction or claim that a failed attack proves privacy; success depends on outputs, overfitting, auxiliary data, and the tested population.

### Fairness metrics are context-dependent and can conflict

- Kleinberg, Mullainathan, and Raghavan [5] formalize calibration within groups and balance conditions for positive and negative classes. They prove that, except for constrained cases such as equal base rates or perfect prediction, these conditions cannot all hold simultaneously. This is evidence that “fairness” is not one universally satisfiable scalar metric; the audit must name the decision setting, protected groups, outcome definition, and selected criterion.
- The theorem is about risk-score assignments and its stated conditions. It does not say every pair of fairness metrics is always incompatible, nor does it choose the policy trade-off. Report subgroup sizes/base rates, uncertainty, thresholding and action costs, and explain why the chosen metric matches the application. Avoid ranking groups by a single noisy metric without context.

### Causal limits of observational prediction

- Pearl [6] separates an observational distribution such as `P(X,Y,Z)` from an intervention query such as `P(Y | do(X))`. A structural causal model combines assumptions, a causal query, and data to derive an estimand; if the query is not identifiable under the assumptions and available data, the inference procedure should report failure rather than turn predictive association into a causal claim. The paper also notes cases where the assumptions have no testable implications, so observational fit cannot validate them.
- The chapter may therefore label model outputs as predictive, associational, or interventional/counterfactual and require an explicit causal graph or design before using causal verbs. This is not the claim that observational data are never useful for causal inference: adjustment can identify some effects under defensible assumptions, and randomized or natural experiments add identification evidence. The limitation is that prediction quality alone supplies neither intervention semantics nor identification.

## Audit design implications

For the grouped evaluation fixture, preserve evidence provenance with every result: model/data version, split and grouping rule, sample counts, metric definition, uncertainty method, and the exact claim the result is intended to support. A compact audit record should include attribution sanity checks; robustness threat model and perturbation budget; privacy unit, mechanism, sensitivity, and composed budget; membership-inference access and operating point; fairness metrics with base rates and context; and a causal-status label with assumptions or design. This keeps a high score from silently becoming a stronger claim than the evidence supports.

## Sources (primary or official)

1. Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. “Sanity Checks for Saliency Maps.” *NeurIPS 2018*. [Official proceedings page](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). Source for model/data randomization checks and limits of visual saliency inspection.
2. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. “Towards Deep Learning Models Resistant to Adversarial Attacks.” *ICLR 2018*. [Original paper on arXiv](https://arxiv.org/abs/1706.06083). Source for robust optimization, adversarial examples, and threat-model-specific guarantees.
3. Cynthia Dwork and Aaron Roth. “The Algorithmic Foundations of Differential Privacy.” *Foundations and Trends in Theoretical Computer Science* 9(3–4), 2014. [Harvard Privacy Tools Project PDF](https://privacytools.seas.harvard.edu/sites/g/files/omnuum6656/files/privacytools/files/the_algorithmic_foundations_of_differential_privacy.pdf). Source for the DP definition, Laplace mechanism, privacy parameters, and composition.
4. Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. “Membership Inference Attacks Against Machine Learning Models.” *IEEE Symposium on Security and Privacy 2017*. [Official IEEE conference PDF](https://www.ieee-security.org/TC/SP2017/papers/313.pdf). Source for black-box membership attacks and empirical leakage.
5. Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. “Inherent Trade-Offs in the Fair Determination of Risk Scores.” *ITCS 2017*. [Original paper on arXiv](https://arxiv.org/abs/1609.05807). Source for incompatibility of calibration and balance criteria under unequal base rates.
6. Judea Pearl. “The Seven Tools of Causal Inference, with Reflections on Machine Learning.” *Communications of the ACM* 62(3), 2019. [UCLA author-hosted paper](https://ftp.cs.ucla.edu/pub/stat_ser/r481.pdf). Source for assumptions/data/query separation, intervention notation, estimands, and non-identifiability limits.

## Cautions for the author

- These sources support evaluation boundaries, not a universal pass/fail score for “responsible” models. Keep normative choices and deployment decisions explicit.
- Do not present attribution heatmaps, robustness percentages, DP noise, attack non-success, subgroup parity, or predictive accuracy as interchangeable evidence. Each answers a different question under stated assumptions.
- The six links were checked as original papers or first-party/official proceedings and repository copies. Claims should be paraphrased; avoid copying paper wording into the lesson.

## Astra High final review (2026-09-08)
Luna High reverified all primary-source URLs and reviewed the reference. Chapter 54 claims were reconciled with executable data, arithmetic, and output. The membership attack knows candidate labels and probabilities, implements no differential privacy, and cannot attribute discrimination causally to memorization. Its input security experiment is a parser rather than an additional HTTP server.

## Consistency unification (2026-09-08)

Local source verification preserved the twelve-row fixture, all measured audit outputs, the member-only 400-step training run at learning rate 0.05, and every validation boundary. The classifier now uses Chapter 4's exact stable sigmoid and binary-cross-entropy primitives and the shared `logit`, `probability`, and `predict` vocabulary. Its `loss(row)` remains intentionally per-example because the membership attack thresholds one candidate's loss; it is not the dataset-mean model loss introduced in Chapter 4.

Chapter 53's actual request body is form-encoded `schema=1&x=<scalar>` and feeds a one-feature regression model. Chapter 54's parser expects comma-separated `schema=1,f0=<number>,f1=<number>` and returns two classifier features. An adapter and a different model input contract are therefore required before the Chapter 53 server can host this audit classifier. No new external claim or source was needed for these code-interface corrections.
