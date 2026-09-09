# Chapter 43: September 2026 redesign verification

Route: one GPT-6 Astra High section owner read the complete original lesson, metadata, research and reference implementations, plus the supplied advanced/interactive/learning-science evidence synthesis. No chapter subauthors or new research delegation. The original source verification below is retained as historical provenance; this current record supersedes old starter/exercise and implementation-scope statements.

New required response masks use telescoping prefix loss/gradient sums through the unchanged decoder API. Full LoRA updates both old factors and verifies direct/merged projection parity. Temperature-two gradient uses coefficients -tau*(p_tau-q_tau) over hard-target gradients; a finite-difference check validates the direct tau-squared objective. Real train, unseen-prefix, generic-retention and distillation-transfer evaluations run. Repeated backwards and full cloned effective models are explicit computational limitations.

The author rechecked the primary PPO and InstructGPT records for the added algorithm/pipeline attribution. New numeric claims come from executable course fixtures, not borrowed benchmark outcomes. See `guidance/redesign/section-08.md` for actual commands, values, prerequisite audit and limits.

## Preserved source-verification history

# Chapter 43 research notes

Route: Sol High author with bounded GPT-5.6 Luna High research. Primary sources verified 2026-09-08.

- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155): supervised demonstrations train an initial instruction-following model before later preference stages. Their dataset and human-preference results are not generalized here.
- Wei et al., [FLAN](https://arxiv.org/abs/2109.01652): instruction templates across task clusters and held-out task evaluation support separating training fit from transfer.
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685): freeze W0, train low-rank factors, initialize one factor randomly and the other to zero, scale by alpha/r. This code uses A[D,r]B[r,V]; the paper's letter ordering differs with orientation.
- Hinton, Vinyals, and Dean, [Distillation](https://arxiv.org/abs/1503.02531): soft teacher probabilities and temperature. The executable stays at τ=1 so its weighted hard-target derivative identity exactly matches the reported KL objective; the lesson derives the τ>1 caveat without claiming to implement it.

Implementation uses the real Chapter 36 decoder for SFT, effective LoRA forward/backward, and both teacher and student. No external code is reused.

## Independent Astra implementation review — 2026-09-08

Route: GPT-6 Astra High owner/reviewer, with fresh bounded GPT-5.6 Luna High primary-source verification (`verify_42_43` for 42–43; `verify_44_46` for 44–46). Earlier Sol High drafts were retained where correct; the Astra owner independently read and corrected all lesson, metadata, reference, starter, and exercise assets.

Verified actual Chapter 36 all-parameter SFT, correctly oriented/scaled output LoRA, simultaneous factor gradients, frozen base, and τ=1 weighted hard-target distillation. Replaced clipped probability KL with stable log-softmax KL and added a direct soft-cross-entropy finite-difference test. Clarified teacher entropy, full-backward memory limitations, and unmeasured student speed.

The Luna High verification checked the original papers and official source URLs listed above. Its concrete findings were integrated by the Astra owner; passing prior author gates was not treated as independent proof. Scoped validation and remaining limits are recorded in `guidance/astra-review-42-46.md`.
