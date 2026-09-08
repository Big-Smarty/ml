# First Principles — Machine learning in Rust

A local, offline-capable course from one neuron to advanced language models. Read the lectures in the website; this README is only repository setup.

```sh
python3 tools/serve.py
```

Open http://127.0.0.1:8000 . Start with **Start here**, then Chapter 1. The website’s **Git guide** explains baseline commits, learning notes, and publishing your history to GitHub. Its **Code guide** at `/conventions.html` explains the common model interfaces, vocabulary, shapes, and loss reductions used throughout the chapters.

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

## Helix and rust-analyzer

Open Helix from the course root, for example:

```sh
helix projects/ch01/starter/src/main.rs
```

The repository's `.helix/languages.toml` links all 114 independent Cargo projects (references, starters, Rustlings, and the optional framework example). The root `rust-toolchain.toml` is the language-server root marker and requests `rust-analyzer` plus `rust-src`, alongside formatting and linting components. This enables navigation into both neighboring chapter libraries and Rust's standard library without combining the course into one Cargo workspace.

If Helix was already open when these settings were added, save your work and reopen it from this directory to load the project configuration. Allow the initial project load to finish. Your existing user-level rust-analyzer settings continue to apply; the project only supplies its root and linked manifests. The settings follow [Helix's project configuration](https://docs.helix-editor.com/master/languages.html#project-and-lsp-root-selection) and [rust-analyzer's linked-project configuration](https://rust-analyzer.github.io/book/configuration#rust-analyzerlinkedprojects).

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

`just consistency` checks the shared neuron signatures, chapter continuity sections, glossary ownership, and source-tagged lesson excerpts. `just build` rebuilds the website. `just check 1 2` builds it and audits content and links; omit chapter numbers for all chapters. `just verify 1 2` also runs the existing reference, starter, and exercise checks. `just verify` checks the whole course and can take substantially longer; it accepts passing learner work and recognizes the supplied intentional TODO failures. Use `just starter-test` and Rustlings to check whether your own exercise is complete; a course validation pass alone does not mean you solved it. These commands reuse `tools/build.py` and `tools/verify.py` rather than maintaining a separate check system.

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
