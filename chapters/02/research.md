# Chapter 2 research

Model route: GPT-5.6 Sol High author with a bounded GPT-5.6 Luna High source-research subagent. The author verified the formulas against the executable Rust implementation and integrated only source claims relevant to this chapter.

- MIT OpenCourseWare's *Calculus* text supports the derivative as a limiting local rate of change and the gradient as the collection of partial derivatives: https://ocw.mit.edu/courses/res-18-001-calculus-fall-2023/mitres_18_001_f17_full_book.pdf
- PyTorch's official gradcheck note documents comparison of analytical Jacobians with finite-difference estimates and the need for tolerances: https://docs.pytorch.org/docs/stable/notes/gradcheck.html
- Rumelhart, Hinton, and Williams (1986) is the original paper used here for historical context on efficient gradient-based weight updates: https://doi.org/10.1038/323533a0
- LeCun et al., *Efficient BackProp*, supports the practical discussion of scaling and learning-rate behavior: https://cseweb.ucsd.edu/~gary/258a/lecun98efficient.pdf

The line data, arithmetic, prose, exercises, and Rust code are course-authored. No source code or figures were reused.


## Astra High review — 2026-09-08

GPT-6 Astra High reviewed every owned lecture, metadata file, research note, reference, starter, and exercise/solution, with bounded GPT-5.6 Luna High primary-source verification (01–07). Verified the complete MSE derivative and finite-difference conventions. Strengthened starter and Rustlings checks with a nonzero model and unequal example contributions so constants and single-row derivatives do not pass.

See [the per-chapter review and validation record](../../guidance/astra-review-01-14.md) for evidence, gate results, and limitations.
