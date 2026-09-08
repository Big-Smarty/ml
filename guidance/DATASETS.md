# Data and provenance

Every reference demo must run on a tiny bundled/course-authored fixture without network. Such fixtures teach mechanisms, not real-world predictive quality. External data preparation is explicit and separate from tests. Record URL, author, license, checksum, expected format, split policy and download size once verified. Never imply a mirror's license is an authoritative license.

- Chapters 1–9: course-authored numeric arrays and XOR. Labels for the opening line follow y=2x+1; learning code receives the samples, not the formula. Hold out inputs never passed to training.
- MNIST: official source https://yann.lecun.org/exdb/mnist/ ; 60,000 training / 10,000 test images, 28×28 grayscale, IDX big-endian headers. Preserve test set; choose validation only from training. Redistribution terms must be verified before bundling external data; provide explicit download instructions and generated IDX fixtures instead.
- UCI Iris https://archive.ics.uci.edu/dataset/53/iris , Wine https://archive.ics.uci.edu/dataset/109/wine , Auto MPG https://archive.ics.uci.edu/dataset/9/auto+mpg . UCI pages list CC BY 4.0. Retain attribution. Define course splits rather than inventing official partitions.
- Language and multimodal smoke tests use original short texts/images/labels included in the course. Real training accepts a user-supplied licensed corpus. Document splitting happens before tokenization training/dedup assessment as appropriate; fit preprocessing only on training data.
- TinyStories is a research reference https://arxiv.org/abs/2305.07759 . Verify the actual dataset card/license for an optional larger corpus; do not bundle an unverified dump.

Network failure, missing data and malformed input must produce actionable errors. Never substitute synthetic data silently when the learner explicitly requests an external dataset. Do not download large corpora while authoring the course.

## Verified MNIST preparation

`python3 tools/prepare_mnist.py --download` explicitly downloads approximately 12 MB from the HTTPS OSSCI mirror listed by the official torchvision MNIST implementation. The original MNIST site timed out during verification; this mirror is labeled as a mirror, not as the dataset author. Archive MD5 values are the dataset fingerprints published in https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py (verified 2026-09-08). MD5 checks accidental corruption/identity against that published list; it is not a cryptographic authenticity claim. Downloads use TLS, bounded sizes, and exact decompressed sizes. The resulting manifest records SHA-256 checksums for raw and derived files.

The helper was executed successfully against all four archives. It selects 500 examples per class from the original training set by a fixed SHA-256 ordering of row indices, giving 55,000 fit rows and 5,000 validation rows. The official 10,000-row test set is retained unchanged. Use fit + validation files during development; use t10k only for final evaluation. Store downloaded data locally under ignored `datasets/downloads/mnist/`; do not bundle it with course distribution. Dataset attribution: Yann LeCun, Corinna Cortes, Christopher J. C. Burges. Dataset terms remain those of the original data, not the software license of the downloader.

`python3 tools/prepare_mnist.py --self-test` checks malformed input, deterministic splitting, class balance, and disjoint/exhaustive membership with generated IDX bytes and no network.
