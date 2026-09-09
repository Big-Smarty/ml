# Arguments are quoted positional values, never interpolated into shell code.
set positional-arguments

help:
    @just --list

alias list := help

# Build the offline guide into dist; never edits learner files.
build:
    @python3 tools/build.py

# Build and serve the guide; optional port defaults to 8000.
serve port="8000":
    @python3 tools/serve.py --port "$1"

# Run a working chapter experiment: just lab 1 [experiment arguments...].
lab chapter *args:
    @python3 tools/chapter.py lab "$@"

alias run := lab

# Check your implementation against the chapter learning goal.
lab-check chapter *args:
    @python3 tools/chapter.py lab-check "$@"

# Check the supplied baseline, plumbing and separate solutions for this section.
lab-test chapter *args:
    @python3 tools/chapter.py lab-test "$@"

alias test := lab-test

# Fetch a section's external dependencies explicitly while online.
deps chapter:
    @python3 tools/chapter.py deps "$@"

# Compile an optimized lab without running it (use before benchmarks).
lab-build chapter:
    @python3 tools/chapter.py lab-build "$@"

# Check the section's Rust code with Clippy, treating warnings as errors.
lint chapter:
    @python3 tools/chapter.py lint "$@"

# Emit optimized assembly for the chapter's section package.
asm chapter:
    @python3 tools/chapter.py asm "$@"

# Run the explained solution; add --check to verify its learning goal.
solution chapter *args:
    @python3 tools/chapter.py solution "$@"

# Format the chapter's section package.
fmt chapter:
    @python3 tools/chapter.py fmt "$@"

fmt-check chapter:
    @python3 tools/chapter.py fmt-check "$@"

# Audit content, metadata, source excerpts and links (authoring check).
check *chapters:
    @python3 tools/chapter.py check "$@"

# Audit the guide and run lab delivery checks, not a learner-completion grade.
verify *chapters:
    @python3 tools/chapter.py verify "$@"

# Explicit hardware execution for the GPU section; see each lab's flags.
gpu chapter *args:
    @python3 tools/chapter.py gpu "$@"

# Explicit real-device tests for chapters 29–32.
gpu-test chapter *args:
    @python3 tools/chapter.py gpu-test "$@"

# Preserved reference code, separate from the new learning labs.
reference chapter *args:
    @python3 tools/chapter.py reference "$@"

reference-test chapter *args:
    @python3 tools/chapter.py reference-test "$@"

# Explicit GPU integration checks for the preserved Chapter 36 decoder.
reference-gpu-test chapter *args:
    @python3 tools/chapter.py reference-gpu-test "$@"

# Help for optional MNIST preparation; --download explicitly opts in.
mnist *args:
    @python3 tools/prepare_mnist.py "$@"

recipes-test:
    @python3 tools/chapter.py self-test

consistency:
    @python3 tools/check_consistency.py
