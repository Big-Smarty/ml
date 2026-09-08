# Chapter 33 research record

Route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High source pass. Sources were checked 2026-09-08 against primary paper or project pages.

- Bengio et al., *A Neural Probabilistic Language Model* (JMLR 2003), https://www.jmlr.org/papers/v3/bengio03a.html — foundational learned distributed representations used to predict the next word.
- Bojanowski et al., *Enriching Word Vectors with Subword Information*, https://arxiv.org/abs/1607.04606 — character n-gram vectors and rare/unseen word motivation.
- Official fastText source, https://github.com/facebookresearch/fastText — primary implementation evidence for UTF-8 and character n-grams.
- Wieting et al., *Charagram*, https://arxiv.org/abs/1607.02789 — count-based character n-gram composition and its order limitations.

The lesson uses original course text and code. It limits claims to mechanism; it does not infer benchmark quality from the tiny run.

The consistency revision required no new external research. It preserves the verified algorithms and fixtures while aligning the private teaching API with the course contract: `Model::loss(data)` reports mean next-byte cross-entropy, `Model::step(data, learning_rate)` performs one update of that objective, and `Model::predict(input)` returns a byte class ID. The lesson now distinguishes class IDs, logits, probabilities, targets, and predictions, and links Chapter 24's canonical broad embedding definition rather than defining a lookup-only duplicate.

Chapter 37's implementation was checked locally: it cleans, deduplicates, and assigns whole-document splits, but does not construct token windows. Chapter 33 therefore places the boundary invariant on the downstream consumer that calls `windows(2)` within each document.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
