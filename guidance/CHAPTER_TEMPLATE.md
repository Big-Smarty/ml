# Reader composition

Use the active AUTHORING.md contract. Structure coherent steps around the learner's question, with enough worked explanation to understand the algorithm. The site supplies short-step and full-chapter views from the same HTML.

A representative first session might ask: What is the sensor getting wrong? What do its errors cost? What happens when we change the weight? Can the program find an improvement? Does it work on a new reading? These are connected questions, not required repeated headings. All details that make a step understandable stay within that step. Link backward for reference, then briefly retrieve the needed idea.

Begin with a concrete input/output problem and visible result. Show real arithmetic before generalized equations. Introduce each symbol, technical word and dimension. Explain what an implementation does using short excerpts; complete source stays linked as reference. Give exact `just lab NN` and `just lab-check NN` commands, learner entry point and what success means. A working baseline is intentionally incomplete as a learning outcome, never broken scaffolding. Supply graduated hints and worked reasoning. Finish each session with a changed input or realistic failure and questions that require explanation.

Native semantic HTML only. Ordered `<section class="learning-step" id="...">` wrappers match metadata steps; each begins with h2. No h1 or external assets. Details/summary for optional help. Tables for calculations, inline SVG for geometry with numerical text equivalent. All concepts remain readable without JavaScript. A focused interactive accompanies only a concept that benefits from changing a variable, not every lesson.
