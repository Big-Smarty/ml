# Chapter 48 research record

## Route and verification

The author commissioned the bounded `research48` subagent using GPT-5.6 Luna High. It was instructed to resolve the top-1 gradient failure, capacity convention, balancing equation, and joint expert/router training from original MoE papers. The author checked the final derivatives numerically away from routing ties.

## Evidence used

The original sparsely gated MoE work describes expert outputs combined by learned sparse gates and joint backpropagation through experts and gating network: <https://arxiv.org/abs/1701.06538>. GShard supplies top-2 routing, normalized selected weights, capacity constraints, and balancing ideas at scale: <https://arxiv.org/abs/2006.16668>. Switch Transformer simplifies dispatch to top-1 while retaining the selected full-softmax gate, and defines an auxiliary loss from hard routing fractions and mean router probabilities: <https://arxiv.org/abs/2101.03961>.

For router probabilities `p = softmax(W_r x)` and selected expert `m`, the reference computes `y = selected_gate(p_m, E_m(x)) = p_m E_m(x)`. It deliberately does not compute `p_m / p_m`. With renormalized top-1, the gate is the constant one while the argmax remains unchanged, so the task output has zero local derivative with respect to every router logit. Keeping `p_m` gives derivative `dL/dz_j = (dL/dy * E_m) p_m (I[j=m]-p_j)`.

The auxiliary term follows Switch: `alpha * E * sum_i f_i P_i`, where `f_i` is the attempted hard-assignment count for expert `i` divided by all T examples before capacity admission and `P_i` is its mean full-softmax probability over the same T examples. Hard frequencies are stop-gradient; only the probability means are differentiated. Capacity admission and dropped-token metrics remain separate. The paper gives the proportional capacity budget. This project explicitly chooses integer capacity `ceil(T/E * capacity_factor)`, at least one; the minimum-one convention is not claimed as a universal Switch rule. Tokens beyond capacity receive no expert task update in this standalone layer.

## Decisions and checks

Three scalar linear experts take two-feature examples. The fixture has three regimes so learned routes can split evenly, but this is constructed rather than discovered structure. `Model::train_epoch` accumulates one complete-batch gradient at the old parameters and performs one optimizer update. Accepted task-gradient sums divide by all T examples, including overflow examples whose task contribution is zero. A central-difference test uses one example away from an argmax boundary and disables balancing; both one router weight and one expert weight match. Another test computes expected overflow from pre-update assignment counts and the exact capacity, and a two-example capacity test checks the all-T denominator directly.

`Model::loss(data)` is explicitly an uncapped inference task-only half-MSE, computed with the named `half_squared_error(prediction, target)` primitive and averaged over every example. This retained objective has a factor of one half and therefore differs from the ordinary MSE introduced in Chapter 1. It excludes auxiliary loss and capacity dropping, which are training mechanics. This CPU fixture does not measure conditional-compute speed, network communication, language-model quality, or scaling behavior.

## Astra High review and correction — 2026-09-08

The user-directed ownership is GPT-6 Astra High, reviewing the existing Sol High draft and implementing corrections. Bounded read-only mathematical/source verification was delegated to GPT-5.6 Luna High (`verify_math`), which browsed original Switch, S4, linear-attention, AEVB, GAN, and DDIM sources. The author integrated its evidence and independently inspected all chapter lecture, metadata, reference code, starters, and Rustlings exercise/solution files.

Confirmed full-softmax selected top-1 gate and stop-gradient hard frequencies against Switch. Added finite-difference verification for every router parameter with the auxiliary loss active, and the explicit two-expert counterexample giving auxiliary value 0.885 below one. Clarified that minimum-one capacity is this project's convention, not a universal paper rule. The starter exposes router logits through the free `softmax(logits)` primitive; the reference's `Model::probabilities(input)` computes logits from `router_weights` and then calls it.
