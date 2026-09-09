# Chapter 46: September 2026 redesign verification

Route: one GPT-6 Astra High section owner read the complete original lesson, metadata, research and reference implementations, plus the supplied advanced/interactive/learning-science evidence synthesis. No chapter subauthors or new research delegation. The original source verification below is retained as historical provenance; this current record supersedes old starter/exercise and implementation-scope statements.

Required learned Bradley–Terry reward model, mean DPO batch gradients and exact verifier policy gradient execute separately. Counterbalanced reward features test held-out equal-length responses; confounded length and opposed preferences expose failures. First-step expected-reward numerics distinguish the objective from chosen-only supervision. InstructGPT supports pipeline distinctions; its full-pipeline results are not attributed to SFT alone.

The author rechecked the primary PPO and InstructGPT records for the added algorithm/pipeline attribution. New numeric claims come from executable course fixtures, not borrowed benchmark outcomes. See `guidance/redesign/section-08.md` for actual commands, values, prerequisite audit and limits.

## Preserved source-verification history

# Chapter 46 research record

## Route and verification

The author commissioned the bounded `research46` subagent using GPT-5.6 Luna High. It was asked for verified primary sources, the DPO objective and gradient, the frozen reference role, and a small independently checkable reward-policy experiment. The author matched its equations to the stable loss and simultaneous updates in the final code.

## Evidence used

Rafailov et al. derive Direct Preference Optimization from KL-regularized reward maximization: <https://arxiv.org/abs/2305.18290>. For prompt `x`, chosen response `y_w`, and rejected response `y_l`, the implemented margin is

`z = beta * [(log pi(y_w|x)-log pi(y_l|x)) - (log pi_ref(y_w|x)-log pi_ref(y_l|x))]`.

The mean loss is `-log sigmoid(z)`. The implementation evaluates it as stable softplus of `-z`, and it computes log-softmax with the maximum-subtraction log-sum-exp identity rather than taking the logarithm of an underflowed probability. `Policy::loss_and_gradient` names the policy and reference log ratios, accepts preference data explicitly, and returns a separate `Gradient` with the policy-logit shape. The reference logits are cloned once and never updated. Gradients from every pair are accumulated at the same old policy, divided by pair count, and applied in one optimizer step; this matters because two pairs share each prompt.

Williams's REINFORCE paper is the primary source for score-function updates: <https://doi.org/10.1007/BF00992696>. Sutton et al. provide the policy-gradient theorem and baseline argument: <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation>. PPO is included only as later practical context, not implemented: <https://arxiv.org/abs/1707.06347>.

## Decisions and checks

The preference model is categorical: each prompt has three complete one-token response choices. This makes the chosen/rejected log-probability calculation real and lets one gradient be checked against central differences without copying an autoregressive decoder. It does not exercise token masks or response-length effects, so the lecture explains the sequence sum separately.

The verifiable-reward experiment parses the course-authored prompt `2 + 3`, evaluates candidate integers 4, 5, and 6, and assigns reward one only to the computed answer. Its objective is the probability-weighted sum over those three candidate rewards for one prompt, not their arithmetic mean. It then follows the exact expected policy gradient rather than using preference pairs. This deliberately separates reward optimization from DPO. Tests include wrong answers and invalid arithmetic text, and the correct candidate reaches probability above 0.99. The checker is transparent and tiny; real verifiers can be incomplete or gamed.

## Consistency revision

The scalar primitive is now consistently named `dpo_pair_loss(policy_logratio, reference_logratio, beta)` in the reference, starter, Rustlings exercise, and solution. The overflow-safe softplus expression and extreme wrong-way log-ratio check are shared across those paths. Optimizer rates use `learning_rate`; `Policy::step` is one optimizer update, whereas Chapter 45's `environment_step` advances an environment. `train_dpo` and `exact_reward_training` remain fixture-specific demo drivers rather than a shared training abstraction.

## Independent Astra implementation review — 2026-09-08

Route: GPT-6 Astra High owner/reviewer, with fresh bounded GPT-5.6 Luna High primary-source verification (`verify_42_43` for 42–43; `verify_44_46` for 44–46). Earlier Sol High drafts were retained where correct; the Astra owner independently read and corrected all lesson, metadata, reference, starter, and exercise assets.

Verified DPO margin, averaged simultaneous pair gradients, frozen reference, stable softplus, and exact verifier policy gradient. Expanded the KL derivation with reward/beta and explained the absence of a hard KL guarantee. Fixed arithmetic overflow rejection and stable log-softmax with large common offsets. Validated all reference rows and made failed updates preserve the old policy; added order, reference, extreme, and invalid-input checks.

The Luna High verification checked the original papers and official source URLs listed above. Its concrete findings were integrated by the Astra owner; passing prior author gates was not treated as independent proof. Scoped validation and remaining limits are recorded in `guidance/astra-review-42-46.md`.
