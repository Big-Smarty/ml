# Chapter 37 research record

Route: GPT-5.6 Sol High author with bounded GPT-5.6 Luna High data-research pass. Verified 2026-09-08.

- Dodge et al., *Documenting Large Webtext Corpora*, https://arxiv.org/abs/2104.08758 — source composition and filtering harms in C4.
- Lee et al., *Deduplicating Training Data Makes Language Models Better*, https://arxiv.org/abs/2107.06499 — duplicates, memorization, and cross-split overlap.
- Soldaini et al., *Dolma*, https://arxiv.org/abs/2402.00159 — transparent large-corpus curation and metadata.
- Magnusson et al., *PALOMA*, https://arxiv.org/abs/2312.10523 — decontamination and bits-per-byte evaluation.
- Gebru et al., *Datasheets for Datasets*, https://arxiv.org/abs/1803.09010 — provenance and intended-use documentation framework.

The project deliberately implements only inspectable whitespace cleaning and small-corpus shingle deduplication. Its CC0 fixture is course-authored.

Consistency boundary: the reference now exposes distinct parsed, cleaned, deduplicated, and split document artifacts. It stops before tokenization and window construction. Global whole-document comparison removes only exact and threshold-matched near duplicates; downstream code must still keep documents separate and add substring or semantic checks when those threats matter.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
