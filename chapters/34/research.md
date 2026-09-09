# Chapter 34 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High research. Verified 2026-09-08.

- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, https://arxiv.org/abs/1508.07909 — original subword BPE procedure.
- Radford et al., *Language Models are Unsupervised Multitask Learners*, https://cdn.openai.com/better-language-models/language-models.pdf — byte-level BPE motivation and tokenizer qualifications.
- OpenAI tiktoken core, https://github.com/openai/tiktoken/blob/main/tiktoken/core.py — primary source that BPE operates on bytes and individual-token text decoding may be lossy.
- RFC 3629, https://www.rfc-editor.org/rfc/rfc3629.html — normative UTF-8 ranges and invalid-sequence rules.

All multilingual strings are short course-authored fixtures. No external corpus is bundled.

Consistency note: this chapter uses `Bpe::fit(data, max_merge_count)` for count-based preprocessing estimation and reserves model `loss`, `step`, and `learning_rate` for the gradient-based roles carried from Chapter 33. `encode(bytes)` returns token IDs; `decode(token_ids)` returns bytes so the round trip is checked before UTF-8 display. The fitter's single byte slice is one document boundary, and vocabulary size is exactly 256 plus the number of learned merge rules.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.

## September 9 redesign

GPT-6 Astra High owns all of section 07 (33–39). Read the complete prior lesson, metadata, research and reference implementations before this redesign. The supplied research synthesis supports worked examples, focused prediction, faded implementation and transfer; these are design inferences, not a measured learning-gain claim for this Rust course. No additional research was needed to settle the stable algorithms.

The new lab is `labs/s07-language-models`. Original projects remain read-only references; active practice is `src/ch34.rs`, with a separate explained solution and an explicit learner goal check. Topic coverage and session-to-step mappings are in meta.json; actual numerical verification and limits are recorded in guidance/redesign/section-07.md. Sources above remain primary-source provenance for inherited technical claims. Historical author routes and validation claims above describe the earlier material, not new runs.
