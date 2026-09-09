# Chapter 38 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High optimization research. Verified 2026-09-08.

- Kingma and Ba, *Adam*, https://arxiv.org/abs/1412.6980 — moments and bias correction.
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, https://arxiv.org/abs/1711.05101 — AdamW update.
- Loshchilov and Hutter, *SGDR*, https://arxiv.org/abs/1608.03983 — cosine schedules.
- PyTorch AMP examples, https://docs.pytorch.org/docs/main/notes/amp_examples.html — accumulation and clipping order.
- PyTorch saving/loading tutorial, https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html — model plus optimizer checkpoint state.

The checkpoint format and optimizer are original. Exact next-step equivalence is tested within the same executable environment; cross-platform bitwise reproducibility is not claimed.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered AdamW, checkpoint continuation, validation, caching, serving, and quantization claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.

## September 9 redesign

GPT-6 Astra High owns all of section 07 (33–39). Read the complete prior lesson, metadata, research and reference implementations before this redesign. The supplied research synthesis supports worked examples, focused prediction, faded implementation and transfer; these are design inferences, not a measured learning-gain claim for this Rust course. No additional research was needed to settle the stable algorithms.

The new lab is `labs/s07-language-models`. Original projects remain read-only references; active practice is `src/ch38.rs`, with a separate explained solution and an explicit learner goal check. Topic coverage and session-to-step mappings are in meta.json; actual numerical verification and limits are recorded in guidance/redesign/section-07.md. Sources above remain primary-source provenance for inherited technical claims. Historical author routes and validation claims above describe the earlier material, not new runs.
