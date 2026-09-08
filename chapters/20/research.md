# Chapter 20 research notes

Route: chapter authored by the assigned high-reasoning author after bounded source verification by `gpt-5.6-luna` at high reasoning. Sources were checked as primary papers or the official dataset page on 2026-09-08. The lecture code and numerical examples are original course material.

- Rumelhart, Hinton, and Williams, “Learning representations by back-propagating errors,” Nature 323 (1986), https://doi.org/10.1038/323533a0. Supports the general claim that hidden weights can be adjusted by backpropagated error derivatives.
- Fukushima, “Neocognitron,” Biological Cybernetics 36 (1980), https://doi.org/10.1007/BF00344251. Supports historical context for hierarchical local pattern recognition and position tolerance; its learning procedure is not the supervised backpropagation used here.
- LeCun et al., “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE 86(11) (1998), https://doi.org/10.1109/5.726791. Supports local receptive fields, shared weights, spatial subsampling, and end-to-end digit recognition.
- LeCun, Cortes, and Burges, official MNIST page, https://yann.lecun.org/exdb/mnist/. Supports the 60,000/10,000 counts, 28×28 grayscale images, and IDX representation. The local preparation workflow uses a documented mirror and preserves a validation split from training; it does not claim new dataset terms.

Claims deliberately avoided: competitive accuracy, shift invariance from architecture alone, or equivalence between the seven-segment fixture and handwriting.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 20–24; the researcher supplied checks and source findings, while Astra implemented the corrections.

Reordered stable cross-entropy subtraction and tested a common 1e16 logit offset. Added valid/truncated/invalid-label IDX cases, finite training checks, and padded asymmetric-kernel starter checks. Actual convolution and full backward remain intact. The Luna researcher independently reran the existing local 2000-fit/2000-validation, one-epoch MNIST command and reproduced loss 2.3018 to 0.4614 and accuracy 10.2% to 86.7%; no download or official test scoring occurred. This was an extra local verification, not a new long training study.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
