# Arguments are passed as quoted positional values, never interpolated into shell code.
set positional-arguments

# List the available course commands (also the default).
help:
    @just --list

alias list := help

# Build the offline course website.
build:
    @python3 tools/build.py

# Build and serve the website on localhost; optional port defaults to 8000.
serve port="8000":
    @python3 tools/serve.py --port "$1"

# Run a reference chapter in release mode: just run 1 [program arguments...].
run chapter *args:
    @python3 tools/chapter.py run "$@"

# Test a reference chapter; remaining arguments go to the Rust test runner.
test chapter *args:
    @python3 tools/chapter.py test "$@"

# Run a learner starter; it should launch before you complete its TODO.
starter chapter *args:
    @python3 tools/chapter.py starter "$@"

# Test a starter; failure at the guided TODO is intentional until you solve it.
starter-test chapter *args:
    @python3 tools/chapter.py starter-test "$@"

# Format one reference chapter.
fmt chapter:
    @python3 tools/chapter.py fmt "$@"

# Check reference formatting without changing files.
fmt-check chapter:
    @python3 tools/chapter.py fmt-check "$@"

# Build and audit content/links; omit chapters to check the whole course.
check *chapters:
    @python3 tools/chapter.py check "$@"

# Build and run existing content/Rust/starter/exercise gates; omit chapters for all.
verify *chapters:
    @python3 tools/chapter.py verify "$@"

# Open the official Rustlings runner, or pass a subcommand such as run ch01_01.
exercises *args:
    @python3 tools/chapter.py exercises "$@"

# Explicit GPU run: chapters 29–32, or chapter 39 with its GPU feature enabled.
gpu chapter *args:
    @python3 tools/chapter.py gpu "$@"

# Explicit ignored hardware tests: chapters 29–32 or 36.
gpu-test chapter *args:
    @python3 tools/chapter.py gpu-test "$@"

# Show MNIST preparation help; downloading requires an explicit --download.
mnist *args:
    @python3 tools/prepare_mnist.py "$@"

# Check chapter validation and literal argument forwarding without running models.
recipes-test:
    @python3 tools/chapter.py self-test

# Check shared API signatures, chapter transitions, glossary ownership, and tagged excerpts.
consistency:
    @python3 tools/check_consistency.py
