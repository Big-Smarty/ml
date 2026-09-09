# Chapter 04 redesign provenance

Author route: one GPT-6 Astra High owner for the complete First principles section, under the approved September2026 redesign. The owner read each original lesson, metadata, research note and preserved reference implementation before creating the new labs. Original reference projects remain untouched.

## Source-grounded content

- [Berkson: Application of the Logistic Function](https://doi.org/10.1080/01621459.1944.10500699): Historical logistic-function source.
- [Penn State Binary Logistic Regression](https://online.stat.psu.edu/stat504/Lesson06): Bernoulli likelihood and logit model.
- [SciPy expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html): Official sigmoid definition.
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): Stable cross-entropy from logits.
- [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression): Probability and classifier framing.

The prior foundations source audit verified these primary-source roles. No course-popularity, ratings or enrollment claim is used as evidence of learning.

## Original contribution and numerical evidence

The sensor fixtures, worked arithmetic, lesson prose, experiment tasks, CLI, code and three browser illustrations are course-authored. New labs reuse narrow mathematical patterns from preserved references, not their old exercise workflow. The section report records executed checks and measured outputs. Baseline runs and ordinary tests pass; explicit learner-goal comparisons truthfully report remaining work. Checks call the learner implementations on inspectable data, including independent variations. They cannot establish that a particular derivation was authored or grade a free-text explanation.

## Learning-design basis and limits

Worked examples precede substantial algorithm changes; learners predict a result, compare a working baseline, implement, inspect discrepancies and transfer to changed inputs. This applies the verified research synthesis on worked examples, retrieval and guided active tasks. It is a design inference for everyday Rust programmers who are new to ML, not a controlled result for this complete curriculum. Short sessions are planning estimates; no navigation or mastery gate is inferred from clicks or baseline success.

## Chapter-specific limits

- Perfect training classes do not establish probability calibration.
- Separable unregularized BCE may keep reducing as weight magnitudes grow.
- Tiny two-feature binary fixtures omit multiclass and real-world label quality.
