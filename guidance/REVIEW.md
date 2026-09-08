# Chapter acceptance and review record

The lead reviews every chapter after author delivery and again in the integrated course. Keep results in `guidance/VALIDATION.md` and machine-readable build checks; do not mark an unrun hardware check as passing.

For each chapter verify: prerequisite order; every promised topic taught; defined symbols/terms; worked arithmetic; implementation matches prose; project is runnable; starter compiles and intended test fails; exercise solution passes; hints and transfer answers exist; source URLs are primary and relevant; limitations are honest. Check all HTML pages for structure, internal links, glossary links, accessibility controls and code downloads. Inspect the actual rendered website at representative viewport sizes and each chapter via rendered navigation/content inspection.

Rust gates: fmt --check; clippy --all-targets -- -D warnings; test. Scope passing gates to reference projects and solved exercises: learner TODOs are intentional and checked separately for expected failure. GPU projects must compile and execute on hardware to earn GPU-verified status. Long capstone training has a reproducible command but is not silently run or reported completed during authoring.
