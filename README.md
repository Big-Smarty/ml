# First Principles — Machine learning in Rust

A local, offline-capable course from one neuron to advanced language models. Read the lectures in the website; this README is only repository setup.

```sh
python3 tools/serve.py
```

Open http://127.0.0.1:8000 . Start with **Start here**, then Chapter 1. The website’s **Git guide** explains baseline commits, learning notes, and publishing your history to GitHub.

```sh
cargo test --manifest-path projects/ch01/Cargo.toml
cargo run --manifest-path projects/ch01/Cargo.toml
```

Small exercises use the official Rustlings community runner:

```sh
cargo install rustlings --version 6.5.0 --locked
cd exercises/cpu
rustlings
```

If the authoring installation exists in `.tools/bin`, use `../../.tools/bin/rustlings` from `exercises/cpu` instead. Chapter `src/` folders are reference projects; `starter/` folders are intentionally unfinished learner projects. Do not run a blanket test over all starters expecting success.

## Short commands with just

The root `justfile` wraps the existing course scripts, Cargo, and the official Rustlings runner. Install [just](https://just.systems/man/en/packages.html) separately if needed; these recipes were checked with just 1.58.0. Run `just` or `just list` to see the commands. Recipes also work from a subdirectory of this repository.

```sh
just serve                       # Build the website and serve localhost:8000
just serve 8080                  # Choose another local port
just run 1                       # Run Chapter 01's reference in release mode
just test 01                     # Test the reference; 1 and 01 both work
just test 1 --nocapture           # Forward arguments to the Rust test runner
just starter 1                   # Run the unfinished learner checkpoint
just starter-test 1              # Expected TODO failure until you solve it
just fmt-check 1                 # Check formatting without changing files
just fmt 1                       # Format this reference chapter
just exercises                   # Open Rustlings in the correct exercise folder
just exercises run ch01_01       # Run one Rustlings exercise
```

`just build` rebuilds the website. `just check 1 2` builds it and audits content and links; omit chapter numbers for all chapters. `just verify 1 2` also runs the existing reference, starter, and exercise checks. `just verify` checks the whole course and can take substantially longer; its validator expects the supplied learner TODOs to fail. For your solved work, use `just starter-test` and Rustlings instead of treating the authoring validator as a learner progress check. These commands reuse `tools/build.py` and `tools/verify.py` rather than maintaining a separate check system.

`run`, `test`, and starter commands use Cargo's offline mode, so install the toolchain and prepare required dependencies first. They do not download datasets. Arguments after the chapter number are passed literally to the program or test runner, including paths containing spaces. For example, `just run 56 generate /tmp/ch56-moe.bin rust 16` uses an existing checkpoint. Quote paths with spaces as usual. Chapter numbers outside 1–56 and path-like chapter arguments are rejected.

GPU execution is explicit: use `just gpu 29` through `just gpu 32`, or `just gpu 39` with any chapter-specific options. `just gpu-test 29` through `just gpu-test 32`, and `just gpu-test 36`, run the ignored hardware tests. These require a compatible hardware Vulkan GPU; ordinary reference tests do not enable them. `just run 29` through `32` directs you to the explicit GPU recipe.

`just mnist` shows preparation help and downloads nothing. `just mnist --self-test` checks the split logic without a network; `just mnist --download` explicitly downloads and prepares MNIST. `just recipes-test` checks chapter validation and argument handling without building or training models. No recipe installs tools automatically or starts extended training without your explicit program arguments.

- `CHAPTERS.md`: chapter map, prerequisites, outcomes, projects and checks.
- `guidance/`: authoring templates, research, numerical/data conventions and validation records.
- `chapters/`: original HTML lecture fragments and glossary metadata.
- `projects/`: independent Cargo reference and starter checkpoints.
- `exercises/cpu/`: Rustlings exercises and solutions; no GPU dependency is required by this pack.
- `site/generated/`: static website assembled by `python3 tools/build.py`.

Build and audit the site with `python3 tools/build.py` and `python3 tools/verify.py`. Add `--rust` for all reference/starter/exercise checks, or `--chapters 01 02 --rust` for selected chapters. GPU execution and extended training have separate explicit commands in their chapters.

All default datasets are tiny course-authored fixtures. Larger data downloads and training runs are explicit. Internet access is needed for initial toolchain/dependency installation and external reading, not for the prepared course website.
