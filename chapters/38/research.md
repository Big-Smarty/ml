# Chapter 38 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High optimization research. Verified 2026-09-08.

- Kingma and Ba, *Adam*, https://arxiv.org/abs/1412.6980 — moments and bias correction.
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, https://arxiv.org/abs/1711.05101 — AdamW update.
- Loshchilov and Hutter, *SGDR*, https://arxiv.org/abs/1608.03983 — cosine schedules.
- PyTorch AMP examples, https://docs.pytorch.org/docs/main/notes/amp_examples.html — accumulation and clipping order.
- PyTorch saving/loading tutorial, https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html — model plus optimizer checkpoint state.

The checkpoint format and optimizer are original. Exact next-step equivalence is tested within the same executable environment; cross-platform bitwise reproducibility is not claimed.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered AdamW, checkpoint continuation, validation, caching, serving, and quantization claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
