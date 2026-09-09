# First Principles — Machine learning in Rust

A course for programmers who are new to machine learning: 56 chapters, nine connected projects, explained numerical examples and working Rust experiments.

Read the guide on the private [First Principles site](https://first-principles-ml.budaniasco.chatgpt.site), or serve it locally:

```sh
just serve
```

Open [localhost:8000](http://127.0.0.1:8000), choose **Setup**, then start Chapter 1. Python uses only its standard library. The build writes `dist/` and never changes your lab files.

## Your first experiment

Install Rust using the [official instructions](https://rust-lang.org/tools/install/) and [just for your system](https://just.systems/man/en/packages.html). Python 3 is needed for the guide and chapter dispatcher. From this repository's root:

```sh
just lab 01
```

The baseline works immediately. Open `labs/s01-foundations/src/ch01.rs`, follow the lesson, and replace the candidate comparison with numerical-gradient training. Check that learning goal separately:

```sh
just lab-check 01
```

A failing goal check reports what your implementation has not yet achieved. Ordinary `just lab-test 01` verifies the supplied baseline, plumbing and completed solutions; it does not grade your learning. After trying the hints, read the separate explained solution under `src/solutions/`:

```sh
just solution 01 --check
```

## Course commands

```sh
just serve                 # Build and open a local server
just lab 01                # Run your chapter experiment
just lab-check 01          # Check your implementation's learning goal
just lab-test 01           # Check the supplied section package
just solution 01 --check   # Verify the explained solution
just fmt 01                # Format that chapter's section package
just lint 01               # Run Rust lint checks
just deps 29               # Fetch GPU-section dependencies while online
just check                 # Audit authored content and site links
just verify                # Full delivery checks for all nine labs
```

`just lab` uses Cargo's offline mode. Chapters with external dependencies require a one-time explicit `just deps NN` while online; replace `NN` with any chapter in that section, for example `just deps 29`. Each section README names any dependencies. No lab command downloads datasets. Default runs use tiny bundled fixtures. GPU execution, larger downloads and extended training require the flags described in the relevant chapter. Start with `just gpu 29` for the explicit GPU path after its setup; missing hardware is reported as unsupported, not a passing hardware test.

`just check` and `just verify` validate the course's delivery. They do not mark your practice complete. The website stores your current step and explicit practice notes in this browser; Rust check records there are self-reported. Progress in the redesigned course starts fresh.

## Files and editor setup

- `labs/`: nine independent Cargo packages, each with chapter experiments and separate solutions.
- `chapters/`: the same authored HTML supports focused steps and a continuous reading view.
- `course.json` and `sections.json`: topic order and section project map.
- `guidance/LEARNING_DESIGN_RESEARCH.md`: cited research, limitations and design decisions.
- `guidance/redesign/`: the 56-chapter objective/explanation/practice/assessment matrix, grouped by section.
- `projects/`: preserved reference implementations, used where they support the new experiments.
- `dist/`: generated site, source viewers and downloadable course archive.

Open your editor from the course root, for example `helix labs/s01-foundations/src/ch01.rs`. The project-local `.helix/languages.toml` links the nine active labs and preserved reference packages for rust-analyzer; it excludes the old exercise runner and starters. The root toolchain requests Rust formatting, linting and language-server components. Reload an already-open editor after changing its project configuration. See [Helix's project language configuration](https://docs.helix-editor.com/languages.html).

The old Rustlings files and starter projects remain in repository history and the original checkout, outside the active guide, commands, editor workflow and download. Existing work on `main` is preserved; this redesign is on `codex/ml-learning-redesign`.

For an offline copy, use the guide's **Download the course** link and unzip it. While online, install Python, just and the requested Rust toolchain, then fetch external dependencies for any sections you plan to use. The archive does not include compilers or Cargo's dependency cache. Once those are available locally, run `just serve` inside the extracted `first-principles` folder. The guide and bundled experiments work offline; external readings and optional dataset downloads need a network.
