# Visual design for a technical learning site: evidence and implementation guidance

**Research date:** 2026-09-09

**Scope:** screen-reading polarity, typography and spacing, visual density and signaling, integration of prose with mathematics, and the relationship between aesthetics and learning.

**Local context inspected:** `site/assets/style.css`, `chapters/01/lesson.html`, and the mathematics contract in `guidance/AUTHORING.md`.

## Executive recommendation

Provide the requested very dark **neutral gray** theme with restrained blue and amber accents, while preserving the site's current behavior: follow the operating-system preference until the reader explicitly chooses a theme, then persist that choice. The dark palette is a valid preference and can be comfortable in some conditions, but the primary evidence does not show that dark mode improves learning. Several controlled studies instead find better proofreading or glance legibility with dark text on a light background, especially at small character sizes. Other work finds lower objective fatigue in some dark-mode conditions, so mode choice and display brightness should remain under the reader's control.

Keep the existing 18 px base size and generous leading, limit long-form prose to roughly 65–72 characters per line, and do not compress navigation, tables, or captions to recover space. The exact line-length optimum is not settled: studies vary with task, scrolling, font, and reader. The range is therefore a defensible starting point, not a biological constant.

Treat prose and mathematics as one reading stream. Author every teaching expression, from inline variables through matrices and worked calculations, as native Presentation MathML; use the same mathematical font treatment inline and in display equations. Display an equation when its structure needs inspection, then explain it immediately beside or below it. Preserve meaningful operator spacing. Controlled instructional studies support spatially integrating mutually referring information, and equation-perception studies show that formally irrelevant visual grouping can change mathematical accuracy. No primary study located here establishes a universal rule that inline mathematics is always better than display mathematics.

Replace a cramped `Operation | Input/output | Meaning` matrix with a roomy semantic definition list: operation name first, signature on its own line, explanation below. Keep real tables for genuine row-by-column comparisons and numeric traces. This is partly a design judgment. Web-layout experiments support consistent grouping and lower density, but no experiment reviewed directly compared this exact API table with this exact definition-list pattern.

Use visual polish to clarify structure: consistent type, predictable components, generous whitespace, and a small number of meaningful accents. Do not add decorative graphics merely to make lessons feel richer. Aesthetics reliably changes first impressions and perceived quality; its effect on measured learning and task performance is mixed. Some emotional-design experiments improve comprehension when **essential** lesson graphics are restyled, while other large experiments find no objective performance effect.

## How to read the recommendations

Each recommendation is labeled by its basis:

- **Measured:** directly supported by a controlled empirical study relevant to the claim.
- **Normative:** required or advised by a published accessibility standard.
- **Design judgment:** a concrete local default derived from the evidence, the current site, and the user's stated preference. It should be validated on the actual course.
- **Indirect:** measured evidence from a related task whose transfer to this course is plausible but not established.

These labels matter. A measured advantage in proofreading is not automatically a measured advantage in delayed ML concept learning. A preference rating is not a comprehension score.

## Current-site diagnosis and implementation status

The inspected stylesheet already has several good reading defaults: an 18 px root, `line-height: 1.7`, visible keyboard focus, a `70ch` lesson column, responsive layout, reduced-motion handling, system-preference-aware light/dark selection, and a persisted theme switch. Chapter 01 generally explains equations adjacent to their first appearance and gives numeric tables meaningful captions.

The style revision has addressed the main local problems. The remaining work is verification against the final build:

1. **The former dark mode was green rather than neutral.** The current implementation corrects this with achromatic surfaces and restrained blue and amber accents. This is a visual-direction decision; the evidence does not identify a uniquely effective palette.
2. **Core reading and navigation text no longer shrink to fit.** Chapter navigation is now 16 px, mobile tables are 16 px, and captions and source listings have 14 px floors. Some auxiliary labels remain smaller; keep them noncritical and verify them under dark polarity and zoom.
3. **The content width now follows the active font.** The current `70ch` cap is a defensible trial value; wide figures, tables, equation containers, and interactive panels still need independent overflow checks.
4. **Mathematics now uses one semantic presentation system.** The course-wide conversion to native Presentation MathML gives inline and display expressions the same mathematical font stack and structured markup. The remaining work is to verify layout and assistive-technology behavior across representative expressions and target browser/AT pairs.
5. **Tables retain readable type while dense API references use definition lists.** Tables now start at `0.9rem` and remain 16 px on narrow screens. The former `Operation | Input/output | Meaning` matrices in the advanced chapters use the semantic `.api-reference` definition-list pattern instead of narrower cells.
6. **Instructional components now differ by structure.** Callouts and warnings use a left rule, worked examples use open horizontal rules, exercises use a contained surface, and equations use their own quiet container. Verify that these distinctions remain clear without relying on color alone.

The completed build and browser checks are recorded in [VISUAL_STYLE_VALIDATION.md](VISUAL_STYLE_VALIDATION.md). Native inline math uses a small scrollable HTML wrapper when necessary; the MathML itself retains its semantics. Display equations and data tables scroll inside their own containers. This avoids shrinking expressions or breaking code identifiers to fit a narrow screen.

## Recommended specification

### 1. Theme and contrast

**Design judgment:** use neutral surfaces, not hue-shifted grays. The implemented dark token set is:

```css
--bg: #111111;
--surface: #191919;
--soft: #222222;
--text: #e7e7e7;
--muted: #b4b4b4;
--line: #3b3b3b;
--control-line: #707070;
--accent: #a5c6f4;
--accent-soft: #252525;
--gold: #ddbc83;
--code: #161616;
```

The backgrounds and text are achromatic; blue and amber have functional roles. This is one usable answer, not an empirically optimal learning palette. The light tokens remain neutral as well: `#fafafa` background, white surfaces, `#252525` text, `#5c5c5c` muted text, `#858585` control boundary, `#285ba8` accent, and `#875915` amber.

Using the WCAG relative-luminance calculation, the important text pairs have substantial contrast:

| Pair | Contrast ratio | Use |
|---|---:|---|
| `#e7e7e7` on `#111111` | 15.27:1 | dark body text |
| `#b4b4b4` on `#111111` | 9.11:1 | dark secondary text |
| `#a5c6f4` on `#111111` | 10.78:1 | dark links and accents |
| `#e7e7e7` on `#191919` | 14.22:1 | text on dark surfaces |
| `#b4b4b4` on `#191919` | 8.48:1 | secondary surface text |
| `#111111` on `#a5c6f4` | 10.78:1 | filled dark-theme button |
| `#707070` on `#191919` | 3.55:1 | dark control boundary |
| `#252525` on `#fafafa` | 14.69:1 | light body text |
| `#5c5c5c` on `#fafafa` | 6.41:1 | light secondary text |
| `#285ba8` on `#fafafa` | 6.38:1 | light links and accents |
| `#858585` on `#ffffff` | 3.69:1 | light control boundary |

**Normative:** WCAG 2.2 requires at least 4.5:1 for ordinary text and 3:1 for large text. Interactive component boundaries and states also need sufficient non-text contrast. See [WCAG 2.2, Success Criterion 1.4.3](https://www.w3.org/TR/WCAG22/#contrast-minimum) and [Success Criterion 1.4.11](https://www.w3.org/TR/WCAG22/#non-text-contrast).

The implemented `--line` is deliberately subtle: `#3b3b3b` is only 1.69:1 against `#111111` and 1.57:1 against `#191919`; the light `#d9d9d9` line is 1.35:1 against `#fafafa`. Those values are suitable for decorative separators whose disappearance loses no information. They are not sufficient as the only cue when a boundary or state is necessary to identify a control. The separate `--control-line` meets 3:1 against adjacent surfaces: dark `#707070` is 3.55:1 against `#191919`, and light `#858585` is 3.69:1 against white. Use it where a boundary is necessary to identify a control, while blue and amber continue to signal focus and selection. Verify each actual control rather than treating one token as universally conforming.

**Measured:** Buchner and Baumgartner found consistently better proofreading under positive polarity across ambient-light and chromaticity manipulations; physiological effort and self-reported strain did not explain the performance difference ([2007, *Ergonomics*](https://doi.org/10.1080/00140130701306413)). Piepenbrock, Mayr, and Buchner varied 8, 10, 12, and 14 pt text in a proofreading task and found that the positive-polarity advantage grew as characters became smaller ([2014, *Human Factors*](https://doi.org/10.1177/0018720813515509)). A separate eye-tracking study found both smaller pupils and better proofreading performance under positive polarity, consistent with a luminance mechanism ([Piepenbrock, Mayr, & Buchner, 2014](https://doi.org/10.1080/00140139.2014.948496)). Dobres, Chahine, and Reimer found worse glance-legibility thresholds for negative polarity in dark ambient illumination, while the polarity difference was small under bright illumination ([2017, *Applied Ergonomics*](https://doi.org/10.1016/j.apergo.2016.11.001)).

**Measured counterweight:** Xie and colleagues varied polarity and six luminance-contrast levels. Their eye measures favored dark mode for visual fatigue, while subjective fatigue and preference favored light mode; higher luminance contrast helped ([2021, *IEEE Access*](https://doi.org/10.1109/ACCESS.2021.3061770)). This does not overturn the proofreading results; it shows that legibility, fatigue, preference, luminance, and ambient conditions are different outcomes.

**Action:** honor the requested dark appearance without forcing it as the universal default. Follow the system preference for readers who have not chosen, retain an obvious theme toggle, persist explicit choice, and avoid making small type carry important information. Do not describe either mode as medically healthier or educationally superior.

**Action:** keep dark surfaces close in luminance so cards do not become a checkerboard. Reserve the brighter blue for links, current navigation, focused controls, and a few instructional signals. Amber can mark warnings and focus. Ordinary headings should usually use the main text color rather than another accent.

### 2. Type size, line length, and spacing

Recommended starting values:

```css
:root { font-size: 18px; }
body { line-height: 1.7; }
.lesson { max-width: 70ch; }
.lesson p { margin-block: 0.75rem 1rem; }
```

Use 18 px on narrow screens too unless a rendered check shows a genuine collision. Keep captions at 14 px or larger and primary chapter navigation at 16 px; smaller auxiliary labels should not carry unique instructional information. Do not use globally condensed tracking for body copy. Keep prose left aligned and ragged right.

**Design judgment:** `70ch` is the center of a 65–72ch trial range, not a universal optimum. Allow code, wide equations, numeric tables, figures, and interactive panels to use a wider breakout container where needed.

The line-length evidence resists a single magic number:

- **Measured:** Dyson and Kipping's two screen-reading experiments found long lines faster than short lines without a comprehension change, while subjective ease did not track performance ([1998, *Visible Language* record](https://eric.ed.gov/?id=EJ573260)).
- **Measured:** Dyson and Haselgrove later found that 55 characters per line supported the best combination of comprehension and speed under normal and fast reading, but the result depended on the reader's goal and scrolling pattern ([2001, *International Journal of Human-Computer Studies*](https://doi.org/10.1006/ijhc.2001.0458)).
- **Measured:** Bernard and colleagues found no adult or child speed/efficiency difference across three online line lengths, though adults preferred shorter measures and rated the medium length best presented ([2003, *Human Factors proceedings*](https://doi.org/10.1177/154193120304701112)).
- **Measured:** Beymer, Russell, and Orton's eye-tracking study of instructional pages found narrow paragraphs slightly faster, with fewer regressions and better post-test retention, but readers were also more likely to abandon the ends of long narrow paragraphs ([2005, INTERACT](https://doi.org/10.1007/11555261_59)).

Together these studies justify avoiding both edge-to-edge desktop prose and a very narrow newspaper column. They do not justify presenting `70ch` as experimentally exact.

**Measured:** In an 82-participant eye-tracking study, 10 pt text produced significantly longer fixation durations than 14 pt, but overall reading speed and retention did not differ significantly; serif versus sans serif also produced no significant eye-tracking or retention difference ([Beymer, Russell, & Orton, 2008](https://doi.org/10.14236/EWIC/HCI2008.23)). This supports keeping text comfortably sized and retaining the installed system sans stack rather than adding a font dependency in search of an unproven learning gain.

**Measured:** Ling and van Schaik compared single, 1.5, and double line spacing in web visual search. Wider spacing improved accuracy and response time; left alignment performed better although participants preferred justification ([2007, *Displays*](https://doi.org/10.1016/j.displa.2007.04.003)). Lonsdale, Dyson, and Reynolds compared whole examination layouts and found faster, more efficient information search with the layout that combined legible type size, line length, leading, margins, and clearer paragraph separation ([2006, *Journal of Research in Reading*](https://doi.org/10.1111/j.1467-9817.2006.00317.x)). Because the latter changes several variables together, it cannot identify one causal CSS property.

**Normative:** WCAG does not require authors to set line height to 1.5. It requires that user overrides up to 1.5 line height, 0.12em letter spacing, 0.16em word spacing, and 2em paragraph spacing cause no loss of content or function ([SC 1.4.12 explanation](https://www.w3.org/WAI/WCAG22/Understanding/text-spacing)). Test that override at every responsive layout.

### 3. Hierarchy, clutter, and signaling

Give each visual signal one stable job:

| Signal | Job |
|---|---|
| Main text color | ordinary explanation and headings |
| Blue accent | links, current location, selected or primary action |
| Amber | warnings and keyboard focus |
| Soft neutral surface | grouped but non-urgent material |
| Left rule | one named class of instructional callout, not every block |
| Whitespace | primary section and paragraph separation |

**Measured:** Mautone and Mayer added organizational and verbal signals such as a preview, headings, and causal pointer words. Across printed text, spoken text, and narrated animation experiments, signaled lessons produced more transfer solutions ([2001, *Journal of Educational Psychology*](https://doi.org/10.1037/0022-0663.93.2.377)). This supports meaningful headings, short previews, and explicit causal transitions. It does not show that colored borders around every paragraph improve learning.

**Measured:** Mayer, Heiser, and Lonn found worse retention and transfer when learners received narration plus concurrent on-screen text, and worse transfer from interesting but irrelevant details, across four short lightning-lesson experiments ([2001, *Journal of Educational Psychology*](https://doi.org/10.1037/0022-0663.93.1.187)). The redundancy result concerns animation, narration, and concurrent text; do not generalize it into a ban on useful written recap in a self-paced text course. The relevant local lesson is to remove duplicated or decorative material that competes at the same moment.

**Measured:** Parush and colleagues manipulated link count, alignment, grouping indications, and density in English and Hebrew web pages. Search performance was particularly poor with many links and variable density and improved when density was uniform ([2005, *Human Factors*](https://doi.org/10.1518/0018720053653785)). This supports a steady rhythm and discourages squeezing some rows while leaving others spacious.

**Indirect:** Rosenholtz, Li, and Nakano developed image-based clutter measures and showed that feature congestion, including color variability, predicted aspects of visual-search difficulty ([2007, *Journal of Vision*](https://doi.org/10.1167/7.2.17)). The work measures search and perception rather than learning. It is useful support for limiting simultaneous color, edge, and orientation changes, not proof that minimal pages always teach better.

**Action:** retain the course's question-led section headings and adjacent explanations. Reduce the number of competing boxed treatments. A warning, an exercise, an equation, and a demo should be distinguishable by label and structure before color. Use whitespace and headings before adding another border, icon, or background.

### 4. Mathematics as part of the prose

The implemented course-wide choice is authored native Presentation MathML. It handles simple inline symbols and complex fractions, roots, scripts, summations, and matrices with the same semantic structure, needs no JavaScript or network renderer, and works when the course is used offline. Keep the CSS attached to `math` and its display container rather than styling plain-text imitations:

```css
math {
  font-family: math, "STIX Two Math", "Cambria Math", serif;
  font-size: 1.08em;
  color: var(--text);
  overflow-wrap: normal;
  word-break: normal;
  letter-spacing: normal;
  max-width: 100%;
  overflow-x: auto;
}

math:not([display="block"]) {
  padding: 0 0.08em;
}

.equation {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 1.15rem 1.35rem;
  overflow-x: auto;
  max-width: 100%;
  font-size: 1.04rem;
}

.equation math[display="block"] {
  margin: 0.2rem 0;
  min-width: max-content;
  text-align: left;
}

math mtext {
  font-family: system-ui, sans-serif;
  font-size: 0.9em;
}
```

For example, the short relation in Chapter 01 belongs inline in the sentence when the sentence reads naturally as one unit:

```html
<math>
  <mrow><mover><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow>
</math>
```

A structured expression that learners need to inspect belongs in the horizontally scrollable display container:

```html
<div class="equation">
  <math display="block">…structured Presentation MathML…</math>
</div>
```

**Platform basis:** MDN classifies the `<math>` element as Baseline/Widely available since January 2023, and its browser-compatibility data records support from Chrome 109, Firefox 4, and Safari 5.1 ([MDN element reference](https://developer.mozilla.org/en-US/docs/Web/MathML/Reference/Element/math), [MDN browser-compatibility data](https://raw.githubusercontent.com/mdn/browser-compat-data/main/mathml/elements/math.json)). Target the interoperable [MathML Core specification](https://www.w3.org/TR/mathml-core/); the specification warns that features outside Core may not interoperate and strongly encourages an OpenType MATH font because ordinary font fallback may render poorly. Avoid `mfenced`, which is outside MathML Core; express fences with `mo` and grouping with `mrow`.

**Accessibility basis and limit:** `<math>` has an implicit `math` role, and Chrome documents exposing MathML through platform accessibility APIs ([Chrome 109 MathML announcement](https://developer.chrome.com/blog/new-in-chrome-109/)). This is not evidence of uniform spoken output. MathJax's accessibility documentation notes substantial variation across browser and screen-reader combinations ([MathJax accessibility documentation](https://docs.mathjax.org/en/v4.0/basic/accessibility.html)). Keep visible prose that defines symbols and explains each formula, and test representative target combinations. Do not add blanket `aria-label` attributes to MathML: WAI warns that naming some containers can hide descendant content from assistive technologies ([WAI naming guidance](https://www.w3.org/WAI/ARIA/apg/practices/names-and-descriptions/)). Use an accessible name only as an intentional fallback whose spoken result has been tested.

Apply these authoring rules:

1. Introduce every symbol in prose before or immediately after its first equation.
2. Use native MathML for every mathematical variable, expression, vector, shape, probability, and worked calculation, including inline mentions. Plain counts and dates remain text; Rust names, types, paths, and literal program output remain `code`.
3. Use a display block for a long loss, multi-step derivation, matrix, or expression that the next paragraph inspects term by term.
4. Keep the explanation adjacent. Do not make the learner shuttle between an equation, a distant legend, and a separate definition table.
5. Encode identifiers and functions with `mi`, numbers with `mn`, operators and fences with `mo`, and structure with Core elements such as `mrow`, `mfrac`, `msub`, `msup`, `msubsup`, `msqrt`, `mover`, `munder`, `munderover`, and `mtable`. Use `mtext` only for prose inside an expression, never as a wrapper for a whole formula. Use upright named functions and the invisible function-application character U+2061 where it clarifies semantics.
6. Keep the same glyph, case, subscript convention, and style for a symbol in prose, equations, figures, and tables.
7. Preserve conventional operator spacing. Do not use CSS `letter-spacing` to force a mathematical expression wider or tighter.
8. Avoid coloring variables solely to communicate correspondence; if color is helpful, repeat the relationship in text or labels.

**Measured:** Tarmizi and Sweller reported five geometry experiments in which conventional worked examples that split attention between sources were no better and sometimes worse than problem solving; integrated formats restored the worked-example advantage ([1988, *Journal of Educational Psychology*](https://doi.org/10.1037/0022-0663.80.4.424)). Chandler and Sweller then reported six electrical-engineering and biology experiments. Integrated instructions helped when separated sources had to be combined to make sense, including during a three-month training study; integration did not help when each source was independently understandable, and nonessential explanations could hurt ([1991, *Cognition and Instruction*](https://doi.org/10.1207/s1532690xci0804_2)). The conditional result is crucial: proximity helps when information must be integrated.

**Measured:** Landy and Goldstone manipulated visually implied groups in four algebraic-equation experiments. Accuracy was highest when perceptual grouping agreed with operator precedence, even though the grouping marks had no formal mathematical meaning ([2007, *Journal of Experimental Psychology: Learning, Memory, and Cognition*](https://doi.org/10.1037/0278-7393.33.4.720)). Jiang, Cooper, and Alibali similarly found that spatial layout affected how 91 undergraduates solved and interpreted expressions containing minus signs ([2014, *Quarterly Journal of Experimental Psychology*](https://doi.org/10.1080/17470218.2014.898669)). These studies support consistent, semantically aligned spacing. They do not establish one universal equation font or block color.

### 5. API references and dense tables

Preserve `<table>` for actual matrices of comparable values: datasets, candidate models, arithmetic traces, and benchmark results. Use `<caption>`, scoped headers, and horizontal scrolling when the relationships really are two-dimensional.

For `Operation | Input/output | Meaning`, use a `<dl class="api-reference">` in which every item has:

```html
<div>
  <dt><code>operation_name</code></dt>
  <dd class="signature">input shape → output shape</dd>
  <dd>One concise sentence stating what the operation computes.</dd>
</div>
```

Recommended presentation:

- one operation per visually separated row;
- at least 0.9–1rem block padding;
- operation name and signature never forced into the same narrow cell;
- signature allowed to wrap naturally;
- one column on narrow screens;
- no reduction in type size below the surrounding instructional text merely to fit;
- optional two-column layout only on wide screens, with the name/signature group on the left and meaning on the right;
- row striping only if scanning repeated rows is difficult after spacing and rules are in place.

**Evidence status:** this transformation is a **design judgment** responsive to the reported cramping. It is consistent with Parush et al.'s measured density result and Chandler and Sweller's conditional integration result, but neither paper tested API references. Treat the learner's ability to match a name, signature, and meaning without horizontal eye travel as the acceptance criterion.

### 6. Aesthetics versus learning

The best-supported claim is modest: visual design changes perception quickly and can affect motivation, but attractive styling alone is not a dependable learning intervention.

**Measured preference/perception:** Tuch and colleagues manipulated visual complexity and prototypicality across 119 website screenshots. Both affected aesthetic ratings after exposures as short as 17–50 ms, and low-complexity, high-prototypicality pages were rated most appealing ([2012, *International Journal of Human-Computer Studies*](https://doi.org/10.1016/j.ijhcs.2012.06.012)). This measures first impressions of screenshots, not comprehension, retention, accessibility, or long-session comfort.

**Measured preference versus retention:** Hall and Hanna assigned 136 participants to four web text/background color combinations. Higher-contrast combinations received better readability ratings; color did not significantly affect quiz retention; chromatic palettes received higher aesthetic ratings ([2004, *Behaviour & Information Technology*](https://doi.org/10.1080/01449290410001669932)). This is especially relevant here: palette preference and measured retention separated in the same experiment.

**Measured learning benefits under specific manipulations:**

- Um and colleagues randomized 118 college students in a multimedia immunization lesson. Restyling intrinsic material with warm colors and rounded, face-like forms induced positive emotion and improved comprehension and transfer in that experiment ([2012, *Journal of Educational Psychology*](https://doi.org/10.1037/a0026609)).
- Plass and colleagues partially replicated the pattern: well-designed material improved positive emotion and comprehension, while transfer effects depended on the shape/color combination ([2014, *Learning and Instruction*](https://doi.org/10.1016/j.learninstruc.2013.02.006)).
- Mayer and Estrella restyled essential virus/cell graphics and found higher post-test scores in two brief college-student experiments, with effect sizes reported as *d* = 0.69 and 0.65 ([2014, *Learning and Instruction*](https://doi.org/10.1016/j.learninstruc.2014.02.004)). The treatment altered the lesson's essential graphics, not merely the page background.

**Measured null or weak objective effects:** Heidig, Müller, and Reichelt assigned 334 students to nine aesthetics/usability conditions. The objective manipulations did not change emotional states; perceived aesthetics related more strongly to motivation than learning outcomes, and the manipulation checks failed ([2015, *Computers in Human Behavior*](https://doi.org/10.1016/j.chb.2014.11.009)). Thielsch, Haines, and Flacke assigned 331 participants to attractive/unattractive interfaces and learning/performance goals. Aesthetics changed perceived aesthetics, content, and usability, but produced no significant accuracy or response-time effect in search, creative, or transfer tasks ([2019, *PeerJ*](https://doi.org/10.7717/peerj.6516)).

**Action:** make the theme pleasant through intrinsic qualities—hierarchy, alignment, spacing, type consistency, coherent surfaces, and predictable components. If an illustration is instructionally necessary, make that essential illustration clear and attractive. Do not add mascots, gradients, ambient animation, or decorative diagrams on the theory that positive emotion will automatically increase learning.

## Priority order for implementation

1. Keep the implemented neutral light/dark tokens and system-preference behavior; persist an explicit reader choice.
2. Use the subtle line token only for decorative separation and the 3:1 control-line token where a boundary is necessary to identify a control.
3. Keep body text at 18 px, mobile tables and primary chapter navigation at 16 px, and the 14 px floor for captions and source listings.
4. Keep the lesson measure around `70ch` while allowing figures, tables, display equations, and demos to manage overflow independently.
5. Maintain the completed course-wide native Presentation MathML conversion, consistent inline and display styling, and validation of MathML Core markup and font fallback.
6. Keep genuine numeric matrices as tables and the cramped API operation/signature/meaning references as definition lists.
7. Preserve the distinct component treatments and assign each visual signal one stable meaning.
8. Render representative pages at desktop, tablet, phone, 200% zoom, and user-overridden text spacing; test representative mathematics with target assistive technologies.

This order addresses the user's stated discomfort first and does not require new dependencies.

## Validation plan for this course

Automated and visual checks can establish implementation quality, not learning efficacy:

- calculate contrast for body text, muted text, links, focus indicators, control boundaries, and text inside filled buttons;
- render Chapter 01 and a later equation-heavy chapter at approximately 390, 768, 1280, and 1600 CSS px widths;
- inspect 200% browser zoom and WCAG text-spacing overrides;
- navigate sidebar, theme control, details, form controls, and demos by keyboard;
- confirm that tables scroll without shrinking text and API definitions wrap without overlap;
- confirm that inline symbols and display equations use consistent glyphs and remain legible in both themes;
- validate authored MathML against the Core subset, inspect complex fractions, scripts, roots, operators, and matrices with the available math-font fallback, and test representative expressions with chosen browser/screen-reader pairs;
- check forced colors/high contrast mode where available.

If the team wants evidence about learning rather than only polish, compare the old and revised presentation with actual learners. Separate outcomes:

1. **Performance:** task time and errors while locating an API operation or interpreting an equation.
2. **Learning:** immediate explanation and a novel transfer question, preferably with a delayed check.
3. **Experience:** comfort, preference, perceived clarity, and willingness to continue.
4. **Context:** theme preference, ambient light, display, viewport, zoom, prior knowledge, and vision correction.

Do not collapse these into one “better design” score. Randomize presentation order where possible, keep lesson content identical, and choose the sample size from a power analysis for the smallest effect worth acting on. A small usability pilot can find severe layout failures but cannot establish that a palette improves learning.

## Evidence limits

- Most polarity studies measure proofreading, lexical decisions, or glance reading rather than sustained technical learning.
- Many screen-typography studies predate current high-density displays and responsive browsers. Their causal comparisons remain useful, but point sizes do not map cleanly to today's CSS pixels or viewing distances.
- Exact line-length findings conflict across tasks and navigation methods. The proposed range is deliberately a trial default.
- Classic cognitive-load experiments often use short science, geometry, biology, or electrical lessons with novices. Their principles transfer plausibly to ML instruction, but that transfer has not been directly tested here.
- Emotional-design studies often use anthropomorphic shapes and warm colors in essential diagrams. They do not show that a dark neutral shell improves comprehension.
- Self-reported comfort, objective eye measures, task performance, and retention can move in different directions.
- Individual vision, age, language fluency, environment, and device matter. Reader choice is part of an evidence-aligned design.
- Native MathML removes a runtime renderer but does not guarantee identical typography or spoken output across fonts, browsers, operating systems, and assistive technologies.

## Primary-source index

The core empirical sources used above are listed here for reproducibility. All are original experiments rather than design-blog summaries.

1. Buchner, A., & Baumgartner, N. (2007). [Text-background polarity affects performance irrespective of ambient illumination and colour contrast](https://doi.org/10.1080/00140130701306413). *Ergonomics, 50*(7), 1036–1063.
2. Piepenbrock, C., Mayr, S., & Buchner, A. (2014). [Positive display polarity is particularly advantageous for small character sizes](https://doi.org/10.1177/0018720813515509). *Human Factors, 56*(5), 942–951.
3. Piepenbrock, C., Mayr, S., & Buchner, A. (2014). [Smaller pupil size and better proofreading performance with positive than with negative polarity displays](https://doi.org/10.1080/00140139.2014.948496). *Ergonomics, 57*(11), 1670–1677.
4. Dobres, J., Chahine, N., & Reimer, B. (2017). [Effects of ambient illumination, contrast polarity, and letter size on text legibility under glance-like reading](https://doi.org/10.1016/j.apergo.2016.11.001). *Applied Ergonomics, 60*, 68–73.
5. Xie, X., Song, F., Liu, Y., Wang, S., & Yu, D. (2021). [Study on the effects of display color mode and luminance contrast on visual fatigue](https://doi.org/10.1109/ACCESS.2021.3061770). *IEEE Access, 9*, 35915–35923.
6. Dyson, M. C., & Kipping, G. J. (1998). [The effects of line length and method of movement on patterns of reading from screen](https://eric.ed.gov/?id=EJ573260). *Visible Language, 32*(2), 150–181.
7. Dyson, M. C., & Haselgrove, M. (2001). [The influence of reading speed and line length on the effectiveness of reading from screen](https://doi.org/10.1006/ijhc.2001.0458). *International Journal of Human-Computer Studies, 54*(4), 585–612.
8. Bernard, M. L., Fernandez, M., Hull, S., & Chaparro, B. S. (2003). [The effects of line length on children and adults' perceived and actual online reading performance](https://doi.org/10.1177/154193120304701112). *Proceedings of the Human Factors and Ergonomics Society Annual Meeting, 47*(11), 1375–1379.
9. Beymer, D., Russell, D. M., & Orton, P. Z. (2005). [Wide vs. narrow paragraphs: an eye tracking analysis](https://doi.org/10.1007/11555261_59). *INTERACT 2005*, 741–752.
10. Beymer, D., Russell, D., & Orton, P. (2008). [An eye tracking study of how font size and type influence online reading](https://doi.org/10.14236/EWIC/HCI2008.23). *Proceedings of the 22nd British HCI Group Annual Conference*, 15–18.
11. Ling, J., & van Schaik, P. (2007). [The influence of line spacing and text alignment on visual search of web pages](https://doi.org/10.1016/j.displa.2007.04.003). *Displays, 28*(2), 60–67.
12. Lonsdale, M. dos S., Dyson, M. C., & Reynolds, L. (2006). [Reading in examination-type situations: the effects of text layout on performance](https://doi.org/10.1111/j.1467-9817.2006.00317.x). *Journal of Research in Reading, 29*(4), 433–453.
13. Parush, A., Shwarts, Y., Shtub, A., & Chandra, M. J. (2005). [The impact of visual layout factors on performance in web pages](https://doi.org/10.1518/0018720053653785). *Human Factors, 47*(1), 141–157.
14. Rosenholtz, R., Li, Y., & Nakano, L. (2007). [Measuring visual clutter](https://doi.org/10.1167/7.2.17). *Journal of Vision, 7*(2), 17.
15. Mautone, P. D., & Mayer, R. E. (2001). [Signaling as a cognitive guide in multimedia learning](https://doi.org/10.1037/0022-0663.93.2.377). *Journal of Educational Psychology, 93*(2), 377–389.
16. Mayer, R. E., Heiser, J., & Lonn, S. (2001). [Cognitive constraints on multimedia learning: when presenting more material results in less understanding](https://doi.org/10.1037/0022-0663.93.1.187). *Journal of Educational Psychology, 93*(1), 187–198.
17. Tarmizi, R. A., & Sweller, J. (1988). [Guidance during mathematical problem solving](https://doi.org/10.1037/0022-0663.80.4.424). *Journal of Educational Psychology, 80*(4), 424–436.
18. Chandler, P., & Sweller, J. (1991). [Cognitive load theory and the format of instruction](https://doi.org/10.1207/s1532690xci0804_2). *Cognition and Instruction, 8*(4), 293–332.
19. Landy, D., & Goldstone, R. L. (2007). [How abstract is symbolic thought?](https://doi.org/10.1037/0278-7393.33.4.720). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 720–733.
20. Jiang, M. J., Cooper, J. L., & Alibali, M. W. (2014). [Spatial factors influence arithmetic performance: the case of the minus sign](https://doi.org/10.1080/17470218.2014.898669). *Quarterly Journal of Experimental Psychology, 67*(8), 1626–1642.
21. Tuch, A. N., Presslaber, E. E., Stöcklin, M., Opwis, K., & Bargas-Avila, J. A. (2012). [The role of visual complexity and prototypicality regarding first impression of websites](https://doi.org/10.1016/j.ijhcs.2012.06.012). *International Journal of Human-Computer Studies, 70*(11), 794–811.
22. Hall, R. H., & Hanna, P. (2004). [The impact of web page text-background colour combinations on readability, retention, aesthetics and behavioural intention](https://doi.org/10.1080/01449290410001669932). *Behaviour & Information Technology, 23*(3), 183–195.
23. Um, E., Plass, J. L., Hayward, E. O., & Homer, B. D. (2012). [Emotional design in multimedia learning](https://doi.org/10.1037/a0026609). *Journal of Educational Psychology, 104*(2), 485–498.
24. Plass, J. L., Heidig, S., Hayward, E. O., Homer, B. D., & Um, E. (2014). [Emotional design in multimedia learning: effects of shape and color on affect and learning](https://doi.org/10.1016/j.learninstruc.2013.02.006). *Learning and Instruction, 29*, 128–140.
25. Mayer, R. E., & Estrella, G. (2014). [Benefits of emotional design in multimedia instruction](https://doi.org/10.1016/j.learninstruc.2014.02.004). *Learning and Instruction, 33*, 12–18.
26. Heidig, S., Müller, J., & Reichelt, M. (2015). [Emotional design in multimedia learning: differentiation on relevant design features and their effects on emotions and learning](https://doi.org/10.1016/j.chb.2014.11.009). *Computers in Human Behavior, 44*, 81–95.
27. Thielsch, M. T., Haines, R., & Flacke, L. (2019). [Experimental investigation on the effects of website aesthetics on user performance in different virtual tasks](https://doi.org/10.7717/peerj.6516). *PeerJ, 7*, e6516.

The normative and platform sources are [WCAG 2.2](https://www.w3.org/TR/WCAG22/), [MathML Core](https://www.w3.org/TR/mathml-core/), the [MDN `<math>` reference](https://developer.mozilla.org/en-US/docs/Web/MathML/Reference/Element/math) and [compatibility data](https://raw.githubusercontent.com/mdn/browser-compat-data/main/mathml/elements/math.json), [Chrome's MathML accessibility implementation note](https://developer.chrome.com/blog/new-in-chrome-109/), [WAI's accessible-name guidance](https://www.w3.org/WAI/ARIA/apg/practices/names-and-descriptions/), and [MathJax's cross-platform accessibility caveat](https://docs.mathjax.org/en/v4.0/basic/accessibility.html).
