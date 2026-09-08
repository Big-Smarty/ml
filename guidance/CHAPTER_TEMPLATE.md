# Chapter template

1. **The question**: a concrete failure or task from the preceding chapter, a visible goal, and why the next concept is necessary. Link prerequisites.
2. **Work through the idea**: plain-language definition, small diagram if useful, numerical example, then general equation with symbols and shapes defined.
3. **Build it in Rust**: explain the data layout, map arithmetic to code, discuss important invariants; link exact complete reference source. The code must implement the described method.
4. **Experiment**: exact root-relative commands, dataset provenance, small deterministic checks, expected quantities (label illustrative output), what changing parameters should do.
5. **Practice**: a tiny Rustlings exercise, a larger starter, progressive hints (concept -> pseudocode -> solution), one realistic bug, one independent variation. Include worked answers, not only instructions.
6. **Review and transfer**: questions answered from memory, explanations in details elements, previous concepts to retrieve in a few days, what this chapter does not establish, next-chapter bridge, primary sources.

Approved classes: `lead`, `callout`, `warning`, `equation`, `worked-example`, `exercise`, `table-wrap`, `caption`. Use semantic tags over decorative wrappers. Shared shell supplies title, chapter number, part, contents rail, previous/next links, source/checkpoint links and completion control. Prose source stays HTML. No embedded external JS/CSS/fonts. For diagrams use inline SVG with title/description and viewBox; CSS handles responsive sizing. Use `<math display="block">` for complex equations or readable text/Unicode for simple equations. Always explain notation in prose too.

An optional interactive demonstration uses vanilla JS in a chapter-owned `demo.js`, loaded by the site builder. It must work without network, have labelled native controls, describe the plotted quantities, and avoid claiming browser visuals execute the learner's Rust code. All required concepts remain understandable without interacting.
