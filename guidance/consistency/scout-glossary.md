# Glossary and terminology scout

Read-only audit of `chapters/*/meta.json` for all 56 chapters, guided by
[`guidance/CONSISTENCY.md`](../CONSISTENCY.md). No chapter, project, or asset
files were changed. The metadata is generally careful; the actions below are
the few changes needed to keep one concept on one definition and to prevent
metrics from being mistaken for objectives.

## Duplicate slugs: keep the earliest generalized definition

Keep one glossary entry for each slug. Remove later chapter-owned entries and
leave their chapter explanations as ordinary prose or link the canonical
entry. The proposed definitions are intentionally broad enough for every
listed use.

| Slug | Keep | Remove | Unified definition and action |
|---|---:|---:|---|
| `logit` | 04 | 10 | **An unrestricted real-valued model score; binary logits are passed through sigmoid and multiclass logits through softmax to obtain probabilities.** Keep the binary log-odds detail in ch04's explanation and the vector/shift/stability details from ch10. Do not call logits probabilities. |
| `data-leakage` | 05 | 13, 19 | **Information from outside the allowed prediction, training, or evaluation boundary that enters fitting, model selection, or evaluation.** Keep future data, repeated entities, preprocessing, and held-out-row cases as examples. |
| `row-major` | 09 | 26 | **A layout in which each matrix row is contiguous and each complete row precedes the next; for `[rows, columns]`, offset `(r,c)` is `r * columns + c`.** Keep ch26's cache-loop explanation as chapter prose. |
| `checkpoint` | 11 | 38 | **A validated snapshot of model state and, when continuation is required, optimizer, progress, random, and data state, saved for later inference or training resume.** Keep ch11's trust-boundary checks; ch38 can explain its versioned training payload without a second definition. Distinguish this from the deployment-oriented `model-artifact` and the recovery operation `checkpoint-recovery`. |
| `teacher-forcing` | 23 | 39 | **Supplying ground-truth previous sequence values or tokens as inputs while predicting shifted targets; it is common in training and, when used for evaluation, should be called teacher-forced evaluation.** State that free-running inference feeds model outputs back as later inputs. Ch23's current “during sequence training or evaluation” wording is too broad without that qualifier. |
| `embedding` | 24 | 33 | **A vector representation of an entity or input, obtained from a lookup table or computed by an encoder.** Keep ch24's learned user/item lookup and ch33's token-ID/address details in explanations; Chapter 52 also computes embeddings from images, so the shared definition must not require a discrete lookup. |
| `recall-at-k` | 24 | 52 | **For each query `q` with nonempty relevant set `R_q`, Recall@k is `|R_q ∩ TopK(q)| / |R_q|`, averaged across queries; with one relevant item it equals hit-rate@k.** Ch24's one-item and multi-item explanation is the right base. Ch52's one-positive fixture currently measures query hit rate; retain the `Recall@1` label only with the one-positive equivalence stated, or report it as `HitRate@1`. |

The only repeated slugs found were the seven above. Do not merge the following
nearby but meaningful concepts: `feature-scaling` (umbrella) and
`standardization` (one method); `mean-squared-error` and `brier-score` (Brier
is a probability-versus-binary-outcome evaluation score); `embedding` and
`representation` (embedding names a vector representation); `checkpoint`,
`model-artifact`, and `checkpoint-recovery` (state, deployment package, and
restore operation); or `data-leakage` and `contamination` (general boundary
violation versus evaluation exposure to items or close variants).

## Semantic boundaries to preserve

### Objective, loss, and metric

The ch01 `loss` definition says “fit a training objective,” while later
chapters correctly report held-out loss and distinguish metrics. Use one
relationship everywhere:

> A loss is the scalar value of a specified objective on named data; it may be
> measured during training or held-out evaluation. An objective may be a data
> loss alone or a composite such as data loss plus regularization or an
> auxiliary term. A metric is any reported evaluation quantity and need not be
> optimized.

Thus ch06/ch39 validation loss is still a loss evaluated without updates;
ch12 Brier score is an evaluation metric, not the classifier's training loss;
ch15 Gini is a split criterion; ch18 inertia and log likelihood are algorithm
 quantities; ch22 reconstruction and InfoNCE losses are separate objectives;
and ch48/ch56 balance terms must remain separate from reported task loss.

### Logits, probabilities, and raw scores

Use `logit`/`logits` for pre-normalization real scores, `probability`/
`probabilities` only after sigmoid or softmax, and `predict` for the resulting
class decision. A recommendation preference, SVM score, PCA projection, and
anomaly score are raw task-specific scores, not logits or probabilities. The
ch31 `logit-loss` entry is a computational form of binary cross-entropy; its
definition should say **“binary cross-entropy evaluated stably from logits”**
so the slug is not read as a generic new loss family.

### Reduction and denominator

Every loss or metric explanation must name the unit and denominator. Preserve
these existing distinctions rather than averaging reported averages:

- ch01–02 MSE is the mean over examples, with no one-half factor; ch04, 07, and 31 use binary cross-entropy, while ch10, 11, and 20 use multiclass cross-entropy. Their reductions are over the units named by each chapter.
- ch09 backward returns gradient sums and its caller owns averaging; ch16's
  regularized objective and ch22's reconstruction/InfoNCE objectives have
  different reductions.
- ch12 Brier is mean over rows; ch17 reconstruction MSE averages row errors
  (not row-coordinate elements), while ch22 averages squared coordinates and
  then rows.
- ch18 inertia and mixture log likelihood are sums; do not compare them as
  mean losses. ch39 weights block means by target-token count. ch48 divides
  accepted task-gradient sums by all examples and reports an uncapped,
  unregularized mean task loss separately.
- ch24 ranking metrics and ch52 retrieval metrics average query-level results;
  the relevant set and whether the quantity is Recall@k or hit-rate must be
  stated.

### Retrieval recall versus hit rate

Classification `recall` is `TP / (TP + FN)`. Retrieval Recall@k counts
retrieved relevant items relative to all relevant items for each query. HitRate@k
counts only whether at least one relevant item appears in the top k. They agree
when every query has exactly one relevant item, as in ch24 and ch52 fixtures;
they diverge with multiple relevant items. Never use “hitrate” as an alias for
general Recall@k.

## Additional real semantic flags

- **ch06 `learning-curve`:** the entry describes loss versus training progress,
  commonly called a training curve; “learning curve” can also mean performance
  versus training-set size. Either rename the slug to `training-curve` or state
  the x-axis explicitly wherever it is used.
- **ch04 `likelihood`:** for continuous observations, likelihood is a density,
  not a probability. The current binary fixture is discrete, but a generalized
  definition should say “probability or density assigned to observed data.”
- **ch12 `brier-score`:** replace “binary probabilities and labels” with
  “predicted probabilities and binary outcomes”; the current phrase can be
  misread as a probability-valued label.
- **ch45 `q-value`:** define it as the expected discounted return starting in a
  state, taking the named action, then following a policy. “Future return” can
  incorrectly exclude the immediate reward; the explanation already includes
  it.

No other metadata definition was found to collapse a specialized algorithmic
quantity into a different one. Keep chapter-local names such as `score`,
`objective`, `inertia`, `responsibility`, `log-probability`, and
`load-balancing-loss` when their owning model and reduction are explicit.
