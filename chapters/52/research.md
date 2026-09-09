## Active redesign route (Section 09)

The active learner route is `just lab 52` → edit `labs/s09-advanced/src/ch52.rs` → `just lab-check 52`. The lesson now contains 2 sessions and 6 ordered question-led steps. Goal: Implement and train symmetric contrastive dual encoders and distinguish narrow retrieval checks from generalization. The separately explained solution is `labs/s09-advanced/src/solutions/ch52.rs`; supplied validation, fixtures, and I/O are in `src/common/ch52.rs`. Original `projects/ch52` implementations remain preserved reference evidence. Earlier workflow descriptions below are historical reference notes and do not replace this active route.

Every session includes a concrete worked calculation, implementation checkpoint, graduated hints, an explained answer, and a changed-input transfer task. The browser illustration in chapter 48 executes only a fixed local arithmetic fixture; no browser control claims to run Rust. Executable goal checks and source-review limits are recorded in `guidance/redesign/section-09.md`.

# Chapter 52 research notes

## Route and verification

Model route: GPT-5.6 Luna High primary-source research. Chapter metadata in `course.json` assigns image/text encoders, contrastive alignment, and retrieval, with a course-authored image-caption-pair fixture. Sources were checked on 2026-09-08. The lesson and Rust implementation should use the sources below for mechanism claims; fixture results must remain teaching checks rather than benchmark evidence.

## Claim-to-source notes

- **Two modality-specific encoders and a shared space.** CLIP trains an image encoder and a text encoder jointly, maps each through a linear projection, and compares the resulting image/text embeddings in one multimodal space. Its pseudocode declares the encoder outputs and projection matrices explicitly. ALIGN independently describes the same dual-encoder pattern with EfficientNet and BERT, joined by cosine similarity. [Radford et al., CLIP, §2.3 and Fig. 3](https://arxiv.org/pdf/2103.00020) · [Jia et al., ALIGN, §4.1](https://arxiv.org/pdf/2102.05918)
- **L2 normalization makes the dot product cosine similarity.** CLIP’s reference pseudocode applies `l2_normalize` to both projected towers before taking the image-by-text matrix product. ALIGN defines `x_i` and `y_j` as normalized embeddings and uses their dot product in both directions. This supports explaining normalization as a scale control, not as a guarantee that the learned semantics are correct. [CLIP, Fig. 3](https://arxiv.org/pdf/2103.00020) · [ALIGN, Eqs. (1–2)](https://arxiv.org/pdf/2102.05918)
- **Symmetric in-batch contrastive loss.** For a batch of `N` paired examples, CLIP treats the diagonal image/text matches as positives and the other `N²−N` pairings as negatives, then averages image-to-text and text-to-image cross-entropy. ALIGN gives the same two directional softmax losses and sums them. This is the basis for a small square similarity matrix and diagonal target labels in the project. [CLIP, §2.3 and Fig. 3](https://arxiv.org/pdf/2103.00020) · [ALIGN, §4.1](https://arxiv.org/pdf/2102.05918)
- **Temperature scales logits and may be learned.** CLIP uses a learned log-parameterized temperature and clips the resulting logit scale to avoid instability; ALIGN calls the temperature `σ`, explains its role when embeddings are L2-normalized, and learns it rather than manually sweeping it. The implementation should keep the temperature positive and avoid asserting that one fixed value is universally optimal. [CLIP, §2.3](https://arxiv.org/pdf/2103.00020) · [ALIGN, §4.1](https://arxiv.org/pdf/2102.05918)
- **Evaluation can be zero-shot classification or cross-modal retrieval.** CLIP forms text embeddings for candidate class descriptions and chooses the highest image/text similarity; the paper evaluates this as zero-shot transfer across more than 30 datasets. Its retrieval appendix reports image-to-text and text-to-image results on Flickr30K and MSCOCO, while noting that performance is weaker than the best image-retrieval systems on MSCOCO. ALIGN evaluates both retrieval directions on Flickr30K and MSCOCO and zero-shot classification on ImageNet variants. LiT further uses zero-shot ImageNet and MSCOCO retrieval to study contrastive alignment after locking a pretrained image tower. [CLIP, §3.1 and App. E.1](https://arxiv.org/pdf/2103.00020) · [ALIGN, §§4.2–4.3](https://arxiv.org/pdf/2102.05918) · [LiT, §§5 and 5.1](https://arxiv.org/pdf/2111.07991)
- **Limitations are material to the chapter’s claims.** CLIP reports weak zero-shot performance on specialized, complex, and counting tasks; its authors also warn that zero-shot evaluation can be an imperfect proxy for task generalization, that the internet image-text data is unfiltered and learns social biases, and that class wording restricts the concepts available at inference. The official model card says deployment is out of scope without in-domain testing, performance depends on class taxonomy, and use should be limited to English because other languages were not purposefully trained or evaluated. [CLIP, §§3.1 and 6–7](https://arxiv.org/pdf/2103.00020) · [OpenAI CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md)

## Sources

1. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” arXiv:2103.00020 (2021). https://arxiv.org/abs/2103.00020
2. Jia et al., “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision” (ALIGN), arXiv:2102.05918 (2021). https://arxiv.org/abs/2102.05918
3. Zhai et al., “LiT: Zero-Shot Transfer with Locked-image text Tuning,” arXiv:2111.07991 (2021). https://arxiv.org/abs/2111.07991
4. OpenAI, “CLIP Model Card” (official source repository). https://github.com/openai/CLIP/blob/main/model-card.md

## Fixture cautions

- The tiny paired fixture is suitable for checking tensor dimensions, normalization, diagonal positives, symmetric loss, temperature behavior, and exact retrieval ranking.
- A small batch supplies few and potentially easy negatives; a low training loss or perfect retrieval can therefore reflect memorization and batch composition rather than useful multimodal alignment.
- Do not report the fixture as evidence for zero-shot transfer, broad retrieval quality, robustness, fairness, multilingual behavior, or production readiness. Those claims require held-out and task-specific evaluation at meaningful scale.
- Keep the executable distinctions explicit: normalized-embedding dot products are raw similarities; division by τ produces training logits; softmax normalizes one declared candidate axis; symmetric cross-entropy averages `2N = 6` row and column terms; Recall@1 evaluates raw-similarity ranks and equals HitRate@1 only because each query has one positive.

## Astra High final review (2026-09-08)
Luna High reverified all primary-source URLs and reviewed the reference. Chapter 52 claims were reconciled with executable data, arithmetic, and output. The perturbation score is a local stability check with the same three concepts; both encoder matrices change and the final model loss is validated before returning.
