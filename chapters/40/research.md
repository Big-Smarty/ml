# Chapter 40 research notes

Route: Sol High author with a bounded GPT-5.6 Luna High research subagent. Sources were checked against original papers or official project documentation on 2026-09-08.

- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180): autoregressive decoding reuses prior K/V states; paged allocation addresses fragmentation. Its reported serving gains and kernel costs are workload-specific, so the lesson makes no speed claim.
- Yu et al., [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling changes the active batch after a one-token model iteration. Reported 36.9× throughput is not transferred to this scalar CPU scheduler.
- Hugging Face [logits processors](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py): temperature divides scores; top-k retains the largest scores before sampling.
- Holtzman et al., [Neural Text Degeneration](https://arxiv.org/abs/1904.09751): fixed candidate truncation and temperature have distribution-dependent quality tradeoffs.
- Hugging Face [TGI streaming](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming): streaming reduces time to first visible output, not necessarily total generation time.

Implementation evidence: Chapter 40 uses Chapter 36's published flat parameter spans and reproduces every inference operator for one new token. The test compares every prefix's full vocabulary-logit row, so this is not an output-only or standalone attention cache.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered AdamW, checkpoint continuation, validation, caching, serving, and quantization claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
