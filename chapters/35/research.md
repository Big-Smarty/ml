# Chapter 35 research record

Route: GPT-5.6 Sol High author, bounded GPT-5.6 Luna High primary-source review. Verified 2026-09-08.

- Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 — scaled dot-product attention, causal masking, and multi-head projections.
- Rumelhart, Hinton, and Williams, *Learning representations by back-propagating errors*, https://doi.org/10.1038/323533a0 — reverse-mode learning foundation.
- PyTorch scaled dot-product attention docs, https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html — current executable mask semantics.
- PyTorch gradcheck docs, https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html — finite-difference comparison and double-precision cautions.

Backward equations were derived independently and verified elementwise in f64. The diagram and tensors are course-authored.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered language-model, tokenizer, attention, normalization, and corpus-audit claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
