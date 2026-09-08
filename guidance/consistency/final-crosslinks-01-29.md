# Final cross-chapter audit: Chapters 01–29

Bounded audit of cross-chapter names, primitive signatures, reductions, and glossary definitions. Chapter files were not modified.

## Actionable issue

- **[P2] Stale Chapter 3 API claim.** [`chapters/04/lesson.html:5`](/home/bigsmarty/Projects/ml/chapters/04/lesson.html:5) says Chapter 3 supplied the method sequence including `numerical_gradient` and then says the sequence is present “in both chapters.” [`projects/ch03/src/main.rs:89`](/home/bigsmarty/Projects/ml/projects/ch03/src/main.rs:89) through line 156 define `predict`, `loss`, `gradient`, `loss_and_gradient`, `step`, and `train`, but no `numerical_gradient`. That test-only finite-difference helper first appears in [`projects/ch04/src/main.rs:106`](/home/bigsmarty/Projects/ml/projects/ch04/src/main.rs:106) and is called by the gradient-check test at line 224. Remove `numerical_gradient` from the Chapter 3/in-both-chapters list, or state explicitly that Chapter 4 adds it as a test-only oracle.

## Checks

- `tools/check_consistency.py` reports no continuity, convention-link, data-source, or canonical-signature finding for Chapters 01–29; its findings are all for later chapters and were excluded.
- All `/code/` links in Chapters 01–29 resolve to existing targets.
- Glossary metadata has no duplicate term slugs in Chapters 01–29.
- Cross-checks of current primitive names, preceding source callers, parameter order, and documented reductions found no additional actionable issue. Model-specific `.loss` methods and mapped tiny exercises remain intentional.

## Root resolution

Corrected the Chapter 4 continuity and implementation description: `numerical_gradient` is test-only here and reuses the Chapters 1–2 technique; it is no longer included in the list carried forward from Chapter 3. No numerical code changed.
