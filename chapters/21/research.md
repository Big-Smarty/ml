# Chapter 21 research notes

Route: chapter authored by the assigned high-reasoning author after bounded source verification by `gpt-5.6-luna` at high reasoning. Verification date: 2026-09-08. The two-block Rust experiment and fixture are original.

- He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016, https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html. Supports the residual formulation F(x)+x, parameter-free identity shortcuts, and the optimization motivation for deep residual networks.
- Ioffe and Szegedy, “Batch Normalization,” ICML 2015, https://proceedings.mlr.press/v37/ioffe15.html. Supports mini-batch normalization as part of a trainable network. The project instead uses per-image normalization and labels that difference everywhere.
- Simard, Steinkraus, and Platt, “Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis,” ICDAR 2003, https://doi.org/10.1109/ICDAR.2003.1227801. Supports the role of image distortions as training augmentation in document recognition.
- Cubuk et al., “AutoAugment,” CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/html/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.html. Supports treating operation, probability, and magnitude as a policy. We do not reproduce policy search.

The validation perturbations are deterministic and disjoint from training arrays, but the set remains a functional smoke test rather than a performance estimate.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 20–24; the researcher supplied checks and source findings, while Astra implemented the corrections.

Checked the two-block chain and normalization derivative. Reordered cross-entropy, checked common large logits and invalid images, and added finite update checks. Corrected metadata to identify the tested first-block kernel gradient and both trained residual blocks. The four validation images remain perturbations of the same source bases.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
