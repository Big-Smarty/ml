# Chapter 45: September 2026 redesign verification

Route: one GPT-6 Astra High section owner read the complete original lesson, metadata, research and reference implementations, plus the supplied advanced/interactive/learning-science evidence synthesis. No chapter subauthors or new research delegation. The original source verification below is retained as historical provenance; this current record supersedes old starter/exercise and implementation-scope statements.

Full bandit/Q-learning/REINFORCE mechanisms are retained. Required PPO now actually collects 32-action batches and reuses each for four clipped-surrogate epochs with frozen collection probabilities. Known expected reward supplies an action-independent baseline; there is no learned critic, GAE or decoder RLHF. Five seeds report expected cumulative bandit regret. arXiv 1707.06347 is PPO, not reward-gaming evidence.

The author rechecked the primary PPO and InstructGPT records for the added algorithm/pipeline attribution. New numeric claims come from executable course fixtures, not borrowed benchmark outcomes. See `guidance/redesign/section-08.md` for actual commands, values, prerequisite audit and limits.

## Preserved source-verification history

# Chapter 45 research record

## Route and verification

The author commissioned the bounded `research45` subagent using GPT-5.6 Luna High. It was instructed to use primary or authoritative sources and return equations, source URLs, implementation guidance, and tiny-fixture limitations. The author checked that the cited pages are the original textbook or papers and reconciled the equations with the final Rust implementation.

## Evidence used

Sutton and Barto's *Reinforcement Learning: An Introduction* (second edition) gives the sample-average bandit estimate and its incremental form, epsilon-greedy action selection, tabular temporal-difference control, and the score-function view of policy gradients: <https://incompleteideas.net/book/bookdraft2018mar21.pdf>. The chapter uses the exact incremental mean `Q <- Q + (R-Q)/N`, including exploration over every action.

Watkins and Dayan's original Q-learning paper defines the off-policy control update and states convergence conditions for the finite tabular setting: <https://doi.org/10.1007/BF00992698>. An action value is the expected discounted return beginning with the immediate reward after a specified state and action, then following a policy. The project uses `reward + discount_factor * best_next_value` for nonterminal transitions and omits the bootstrap at the terminal state. Its constant learning rate, decaying exploration, and finite 600 episodes are explicitly outside the theorem's full conditions.

Williams's REINFORCE paper supplies the stochastic log-probability gradient estimator: <https://doi.org/10.1007/BF00992696>. Sutton et al. formalize policy gradients with function approximation and the role of baselines: <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation>. For two softmax actions, the implementation maps two unrestricted policy logits to two normalized probabilities, uses `one_hot(action) - probabilities` as the derivative of the selected action's log probability, and subtracts a moving reward baseline. This one-step environment makes the observed reward the complete return for each episode.

## Decisions and checks

The three mechanisms stay separate so a learner can see what each adds. The bandit has no state transition. The five-state chain introduces delayed value propagation with a lookup table. The REINFORCE experiment directly parameterizes a policy and obtains its update from sampled rewards; it is not a disguised value update.

The deterministic generator makes output reproducible but is not cryptographic and is not evidence about behavior across seeds. Tests check that the best bandit arm is identified, all nonterminal grid states prefer the short path, and the higher-reward policy action exceeds probability 0.9. The result is a mechanics check on a stationary simulator. It does not test function approximation, continuous actions, nonstationarity, safety constraints, or difficult exploration.

## Independent Astra implementation review — 2026-09-08

Route: GPT-6 Astra High owner/reviewer, with fresh bounded GPT-5.6 Luna High primary-source verification (`verify_42_43` for 42–43; `verify_44_46` for 44–46). Earlier Sol High drafts were retained where correct; the Astra owner independently read and corrected all lesson, metadata, reference, starter, and exercise assets.

Verified sample-average bandits, epsilon exploration, terminal Q-learning targets, and sampled REINFORCE with the pre-update action-independent baseline. Corrected the four-step worked return to 0.800325. Added absorbing terminal behavior, sentinel-bootstrap tests and exact learned path values. Clarified return-to-go and baseline ordering.

The Luna High verification checked the original papers and official source URLs listed above. Its concrete findings were integrated by the Astra owner; passing prior author gates was not treated as independent proof. Scoped validation and remaining limits are recorded in `guidance/astra-review-42-46.md`.
