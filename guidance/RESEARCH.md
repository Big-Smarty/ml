# Research basis and curriculum decisions

Initial research used six independently scoped GPT-5.6 Luna High agents: tutorial formats, learning design, ML foundations, Rust CPU systems, GPU systems, and language models. The lead synthesized their findings. Sol High authors produced the initial chapter drafts with Luna High research. The user subsequently assigned Chapters 42–56 and all-chapter review/correction to Astra High, retaining Luna High or Max for bounded research and technical verification. Chapter research records and Astra review reports preserve that provenance.

## Learning design

- Rustlings supports community exercises, ordinary Rust tests, hints, solutions and watch mode: https://rustlings.rust-lang.org/community-exercises/ and https://rustlings.rust-lang.org/usage/ . Reuse that runner rather than invent another.
- Ziglings: https://github.com/ratfactor/ziglings (official migration notice points to Codeberg). Tiny repairs provide fast feedback but do not replace larger projects.
- Haskellings: https://github.com/MondayMorningHaskell/haskellings . Distinguishes compile, test and executable exercises.
- Vulkan Guide: https://vkguide.dev/ and https://vkguide.dev/docs/new_chapter_0/building_project/ . Cumulative projects and reference checkpoints; it assumes graphics prerequisites, unlike this course.
- Vulkan Tutorial: https://vulkan-tutorial.com/ . Its own introduction now warns implementation guidance is outdated. Borrow pedagogical structure, use current Khronos documents for technical claims.
- LearnOpenGL: https://learnopengl.com/Introduction . Readable nested navigation and linked technical references. Write original prose and visuals rather than copying licensed material.
- Roediger & Karpicke (2006), retrieval practice: https://pubmed.ncbi.nlm.nih.gov/16507066/ . Delayed recall motivates closed-book questions before answer reveals.
- Cepeda et al. (2006), distributed practice: https://pubmed.ncbi.nlm.nih.gov/16719566/ . Spacing depends on desired retention; suggested 1/3/7/14-day reviews are a practical default, not a universally optimal schedule.
- Renkl et al. (2004), fading worked examples: https://doi.org/10.1023/B:TRUC.0000021815.74806.f6 . Move from full example to completion to independent work.
- Rohrer & Taylor (2007), interleaving mathematics: https://digitalcommons.usf.edu/psy_facpub/1767/ . Mix old and new problem types after initial focused practice.
- These studies do not directly validate a Rust ML course; curriculum choices are evidence-informed extrapolations.

## Coverage and systems

Stanford CS229 https://cs229.stanford.edu/ and Google's ML Crash Course https://developers.google.com/machine-learning/crash-course/ establish the need for statistics, classical models, data discipline and production thinking alongside networks. MIT 18.06 https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/ supports just-in-time linear algebra. Nielsen https://neuralnetworksanddeeplearning.com/ and micrograd https://github.com/karpathy/micrograd provide useful conceptual comparisons, not dependencies.

Stable SIMD uses std::arch and runtime detection: https://doc.rust-lang.org/stable/std/arch/ . Portable std::simd is currently nightly-only: https://doc.rust-lang.org/nightly/std/simd/ . Benchmark principles: https://llvm.org/docs/Benchmarking.html . GPU path: https://docs.rs/wgpu/latest/wgpu/ and https://docs.vulkan.org/guide/latest/compute_shaders.html . Query actual features rather than assuming capabilities from a GPU marketing name.

LLM spine: https://cs336.stanford.edu/ , https://github.com/karpathy/llm.c and https://github.com/karpathy/llama2.c . Attention https://arxiv.org/abs/1706.03762 ; FlashAttention https://arxiv.org/abs/2205.14135 ; LoRA https://arxiv.org/abs/2106.09685 ; Switch Transformers https://arxiv.org/abs/2101.03961 ; Mamba https://arxiv.org/abs/2312.00752 . Learn each mechanism on a tiny model before discussing scale.

## Compute boundary

Target: Ryzen 9 9900X, about 30 GiB host RAM, Radeon RX 6950 XT. The initial sandbox could not expose `/dev/dri`; an approved host-level probe resolved device access and actual kernels passed on RADV/Mesa 26.2.2, Vulkan 1.4.354. The decoder's GPU logits, gradients, and updates match its CPU reference. The exact 14,442,496-parameter configuration completed bounded CPU and GPU updates, including CPU checkpoint resume. See `VALIDATION.md` for evidence and measurement limits.

A roughly 15M-parameter model is an educational training target, not a promise of general-purpose assistant quality. Estimate time only after measuring effective throughput. Small deterministic verification runs are separate from extended training, which was not performed during course generation.

## Offline syntax highlighting

The reader bundles PrismJS 1.30.0 under its MIT license. `site/assets/prism.js` concatenates the official minified components in this order: core, clike, rust, wgsl, bash, toml, python, javascript. All were retrieved from `https://raw.githubusercontent.com/PrismJS/prism/v1.30.0/components/prism-NAME.min.js`; the release's `LICENSE` is preserved as `site/assets/prism-LICENSE.txt`. Reference: https://github.com/PrismJS/prism/releases/tag/v1.30.0 . Luna High verified the official Rust/WGSL component support; the lead checked text preservation and browser integration. No CDN or network fetch is used for highlighting. Explicit language labels avoid guessing that terminal output is code. Token colors are in the course stylesheet and preserve the dark code surface in either theme.
