# Chapter 44: September 2026 redesign verification

Route: one GPT-6 Astra High section owner read the complete original lesson, metadata, research and reference implementations, plus the supplied advanced/interactive/learning-science evidence synthesis. No chapter subauthors or new research delegation. The original source verification below is retained as historical provenance; this current record supersedes old starter/exercise and implementation-scope statements.

BM25, trained dual word-average embeddings, exact inner-product search, source text extraction and distinct coverage/risk denominators execute. The fourth query France lunar chess champion actually passes the heuristic gate despite being unsupported: source provenance does not establish relevance. No generative LM answering is claimed.

The author rechecked the primary PPO and InstructGPT records for the added algorithm/pipeline attribution. New numeric claims come from executable course fixtures, not borrowed benchmark outcomes. See `guidance/redesign/section-08.md` for actual commands, values, prerequisite audit and limits.

## Preserved source-verification history

# Chapter 44 research notes

Route: Sol High author with bounded GPT-5.6 Luna High research. Primary sources verified 2026-09-08.

- Robertson and Zaragoza, [BM25 and beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf): IDF, saturating term frequency, length normalization, and corpus-dependent parameters.
- Karpukhin et al., [DPR](https://www.cs.princeton.edu/~danqic/papers/emnlp2020a.pdf): separately encoded query/passages, dot-product retrieval, supervised positives and negatives. Reported benchmark gains are not generalized.
- [Faiss exact index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes): flat inner-product/L2 indexes perform exhaustive exact search; approximate indexes trade recall for cost.
- Lewis et al., [RAG](https://proceedings.nips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf): retrieval combined with a parametric generator; generative and extractive answering are distinguished.
- Gao et al., [ALCE](https://aclanthology.org/2023.emnlp-main.398/): answer correctness, citation completeness, and citation entailment are separate.
- Kamath et al., [Selective QA](https://aclanthology.org/2020.acl-main.503/): abstention must be evaluated with coverage and can fail under domain shift.

The implementation is an actual retrieval-augmented extractive pipeline. It intentionally does not label its answerer as LM generation. A test changes retrieved fact text and observes the answer change without retraining, proving dependence on external evidence.

## Independent Astra implementation review — 2026-09-08

Route: GPT-6 Astra High owner/reviewer, with fresh bounded GPT-5.6 Luna High primary-source verification (`verify_42_43` for 42–43; `verify_44_46` for 44–46). Earlier Sol High drafts were retained where correct; the Astra owner independently read and corrected all lesson, metadata, reference, starter, and exercise assets.

Verified BM25 variant, dual word-average encoder gradients, exhaustive inner-product ranking, and extraction from retrieved fact text. Added central differences for both embedding tables and non-first/negative-score exact-search exercise cases. Clarified zero-score lexical ties and overlap-based abstention failures. No generative language-model answering is claimed.

The Luna High verification checked the original papers and official source URLs listed above. Its concrete findings were integrated by the Astra owner; passing prior author gates was not treated as independent proof. Scoped validation and remaining limits are recorded in `guidance/astra-review-42-46.md`.
