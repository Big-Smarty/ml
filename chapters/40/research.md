Final scaffold refinement: named embedding, QKV append and attention exercises use supplied scalar operators, residual/FFN blocks, vocabulary projection and incremental orchestration. The independent `40 --stages` fixtures expose each subgoal before replacing the runnable full-prefix dispatch. The solution passes every stage and the full six-prefix cache goal.

# Chapter 40: September 2026 redesign verification

Route: one GPT-6 Astra High section owner read the complete original lesson, metadata, research and reference implementations, plus the supplied advanced/interactive/learning-science evidence synthesis. No chapter subauthors or new research delegation. The original source verification below is retained as historical provenance; this current record supersedes old starter/exercise and implementation-scope statements.

Actual Chapter 36 decoder dependency; all-prefix equality, real per-layer cache contents and processed projection rows. Sampler and request event order are executable. Browser work/byte counts are illustrations, not latency.

The author rechecked the primary PPO and InstructGPT records for the added algorithm/pipeline attribution. New numeric claims come from executable course fixtures, not borrowed benchmark outcomes. See `guidance/redesign/section-08.md` for actual commands, values, prerequisite audit and limits.

## Preserved source-verification history

# Chapter 40 research notes

Route: Sol High author with a bounded GPT-5.6 Luna High research subagent. Sources were checked against original papers or official project documentation on 2026-09-08.

- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180): autoregressive decoding reuses prior K/V states; paged allocation addresses fragmentation. Its reported serving gains and kernel costs are workload-specific, so the lesson makes no speed claim.
- Yu et al., [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling changes the active batch after a one-token model iteration. Reported 36.9× throughput is not transferred to this scalar CPU scheduler.
- Hugging Face [logits processors](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py): temperature divides scores; top-k retains the largest scores before sampling.
- Holtzman et al., [Neural Text Degeneration](https://arxiv.org/abs/1904.09751): fixed candidate truncation and temperature have distribution-dependent quality tradeoffs.
- Hugging Face [TGI streaming](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming): streaming reduces time to first visible output, not necessarily total generation time.

Implementation evidence: Chapter 40 uses Chapter 36's published flat parameter spans and reproduces every inference operator for one new token. The test compares every prefix's full vocabulary-logit row, so this is not an output-only or standalone attention cache.

Consistency revision: the assigned GPT-5.6 Sol High author rechecked the complete lesson, reference, starter, exercise, solution, and metadata against the Chapter 36 and 39 interfaces. The earlier bounded GPT-5.6 Luna High primary-source review remains the source check for caching and serving claims; no new online research was needed for terminology and interface unification.
