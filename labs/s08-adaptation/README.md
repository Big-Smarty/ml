# Efficient and adapted LLMs: a controlled experiment lab

Seven chapter entries share one CLI and the preserved Chapter 36 decoder. All data are course-authored, tiny, deterministic, offline fixtures. No model download, service, GPU or paid compute is required.

```bash
just lab 40
just lab-check 40
just solution 40
just solution 40 --check
just lab-test 40
```

Replace 40 with any chapter 40–46. `just lab NN` and `just lab-check NN` are root shortcuts. `src/chNN.rs` contains your working baseline and readable goal check. `src/solutions/chNN.rs` is a separate completed implementation with reasoning comments and numerical tests. Baseline runs and `just lab-test 40` pass on delivery. Learner `--check` intentionally reports unmet learning goals until you implement them; it does not deliberately crash the baseline or make ordinary tests fail.

| Chapter | Useful supplied baseline | Core implementation | Evidence |
|---|---|---|---|
| 40 | Full-prefix decoder, greedy choice, serial requests | Per-layer incremental decoder, temperature/top-k sampling, round robin | Every-prefix logits, actual projection-row work/cache contents, three-request event order |
| 41 | Global-scale int8 storage and dequantizing matvec | Per-row int8 and packed int4, sign extension, error/storage accounting | Odd tails, negative codes, unseen 4×9 matvec, real decoder projection comparison |
| 42 | Exact causal scalar GQA | Online state, tile merging, bounded causal output | Uneven tiles, actual peak score count, increasing maxima, oracle agreement |
| 43 | Full SFT, fixed-A linear adapter, tau=1 teacher mixtures | Response-mask selection, both LoRA factor gradients, tau=2 soft-target gradient | Masked-target invariance, frozen base, finite differences, actual train/held-out runs |
| 44 | Lexical overlap, supplied trained two-tower retriever, L2 search, unconditional error rate | BM25, exact inner-product search, source context and coverage/risk | Length counterexample, negative scores, trained paraphrase retrieval, unsupported overlap |
| 45 | Round-robin bandit, myopic reward table, supplied REINFORCE | Epsilon-greedy collection, full Q-learning, clipped PPO batch training | Exact chain returns, sampled policy training, five-seed regret, both clip signs |
| 46 | Chosen-only supervision, feature-average reward score, verifier-selected target | Learned reward model, stable DPO batch gradient, exact expected-reward gradient | Central difference, batch order, frozen reference, counterbalanced reward transfer, first-step reward derivative |

Preserve all supplied shape/input checks. Complete the named algorithm stages while preserving validation and expected thresholds. Printed work counters count actual executed rows/allocated score slots; do not change a counter to conceal work. Keep an experiment record: hypothesis, fixed inputs/seed, baseline, one intervention, invariant, measured result, interpretation, unfamiliar case and limitation. The HTML chapters provide session-sized checkpoints, progressive hints and explanations.

The decoder uses f32 and row-major `[in,out]` projection spans; quantized standalone weights use `[out,in]` with an explicit transpose at the decoder boundary. The small RL/preference policies use f64. Tolerances are fixture-specific. Distillation's repeated hard-target backwards and masked prefix differences favor readable mathematics over training efficiency. The LoRA reference clones an effective decoder and computes a full backward; trainable parameter count is not a measured memory saving.

Chapters 41 and 42 accept `--bench` after the chapter number. These opt-in release measurements cover allocating end-to-end CPU calls; record CPU model, compiler and command with median/range, and do not call them allocation-free kernel benchmarks. Weight compression and attention recurrence establish mechanics, not universal speedups. The retrieval answerer extracts supplied text; it does not run an LM generator. PPO, DPO and learned reward models are tiny policies, not decoder RLHF or alignment evidence.

Chapter 40 additionally accepts `40 --stages` and `40 --solution --stages`. Four independent diagnostics expose learned-position selection, K/V append preservation, cached attention and incremental-prefix parity. They run to completion and print each stage status, so an unfinished early stage does not hide later work. Supplied scalar operators, residual/feed-forward branches, output projection and incremental orchestration keep the learner edit bounded; once these stages pass, route `step` to `incremental_step`. The ordinary full-prefix baseline stays runnable during these edits.
