# Chapter 39 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High capstone research. Verified 2026-09-08.

- Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 — causal decoder and teacher-forced sequence training.
- nanoGPT training source, https://github.com/karpathy/nanoGPT/blob/master/train.py — compact next-token training, evaluation, accumulation, scheduling, and checkpoint workflow.
- nanoGPT model source, https://github.com/karpathy/nanoGPT/blob/master/model.py — causal mask, context cropping, and autoregressive sampling.
- PyTorch CrossEntropyLoss docs, https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html — token-average categorical loss semantics.
- Creative Commons CC0, https://creativecommons.org/publicdomain/zero/1.0/ — terms applied by the course authors to the bundled training and validation prose.

The Rust implementation and data are original. Exact parameter counts come from checked code and tests. Timing output is measured at run time; no throughput or generation-quality figure is fabricated.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered AdamW, checkpoint continuation, validation, caching, serving, and quantization claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.

## September 9 redesign

GPT-6 Astra High owns all of section 07 (33–39). Read the complete prior lesson, metadata, research and reference implementations before this redesign. The supplied research synthesis supports worked examples, focused prediction, faded implementation and transfer; these are design inferences, not a measured learning-gain claim for this Rust course. No additional research was needed to settle the stable algorithms.

The new lab is `labs/s07-language-models`. Original projects remain read-only references; active practice is `src/ch39.rs`, with a separate explained solution and an explicit learner goal check. Topic coverage and session-to-step mappings are in meta.json; actual numerical verification and limits are recorded in guidance/redesign/section-07.md. Sources above remain primary-source provenance for inherited technical claims. Historical author routes and validation claims above describe the earlier material, not new runs.
