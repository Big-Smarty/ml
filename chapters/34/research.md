# Chapter 34 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High research. Verified 2026-09-08.

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, https://arxiv.org/abs/1508.07909 — original subword BPE procedure.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, https://cdn.openai.com/better-language-models/language-models.pdf — byte-level BPE motivation and tokenizer qualifications.
- OpenAI tiktoken core, https://github.com/openai/tiktoken/blob/main/tiktoken/core.py — primary source that BPE operates on bytes and individual-token text decoding may be lossy.
- RFC 3629, https://www.rfc-editor.org/rfc/rfc3629.html — normative UTF-8 ranges and invalid-sequence rules.

All multilingual strings are short course-authored fixtures. No external corpus is bundled.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
