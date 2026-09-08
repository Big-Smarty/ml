# Chapter 36 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High architecture research. Verified 2026-09-08.

- Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 — decoder blocks, residuals, attention, feed-forward layers.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, https://cdn.openai.com/better-language-models/language-models.pdf — GPT-2 pre-normalization and final normalization.
- Ba, Kiros, and Hinton, *Layer Normalization*, https://arxiv.org/abs/1607.06450 — per-example feature statistics and learned gain/bias.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, https://arxiv.org/abs/2002.04745 — pre-LN/Post-LN gradient behavior.

The implementation is original and uses tensor-level manual backprop. No ML, tensor, BLAS, or autodiff framework is used. The GPU bridge calls the chapter 30 primary course API and names per-operation transfer overhead.

Astra High audit with Luna High technical verification: added the original [GELU paper](https://arxiv.org/abs/1606.08415), checked the implemented tanh derivative, target-shift contract, full parameter-count formula, and large-offset stable loss grouping.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
