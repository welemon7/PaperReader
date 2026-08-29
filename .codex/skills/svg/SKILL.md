---
name: svg
description: "Generate a polished, self-contained SVG that visualizes supplied content inside the exact harness-provided fill area. Use for browser-renderable diagrams, flows, timelines, comparisons, charts, architecture views, and technical illustrations."
---

# Content-to-SVG Renderer

Generate the SVG itself from the user's content and the available block dimensions. Treat this file as an execution contract, not as a tutorial about SVG libraries.

## Output contract

- Return only one complete, valid SVG document. Do not return Markdown fences, prose, JSON, XML comments, or multiple alternatives.
- Use the exact supplied `width` and `height` on the root `<svg>`. Preserve the supplied aspect ratio. Never choose a default canvas when dimensions are available.
- Set `viewBox="0 0 width height"` using the same numeric coordinate space as the dimensions. Use `preserveAspectRatio="xMidYMid meet"` unless the harness explicitly requires stretching.
- Keep every visible mark, stroke, label, marker, and shadow inside the viewBox. Account for stroke width when placing objects near edges.
- Make the result self-contained: no external images, fonts, stylesheets, scripts, links, or runtime dependencies. Use basic SVG 1.1 elements and inline styles.
- Use ASCII in generated markup unless the supplied content requires another script. Escape `&`, `<`, `>`, and quotes correctly in text and attributes.
- Add a concise `<title>` and `role="img"` when the visual has meaningful standalone content.

## Inputs and sizing

The harness supplies content and the exact usable region after removing invalid blank space. Interpret the input as:

```text
CONTENT: facts, labels, relationships, numbers, or narrative to visualize
WIDTH: exact usable width in CSS pixels
HEIGHT: exact usable height in CSS pixels
OPTIONAL: requested style, palette, emphasis, or semantic constraints
```

Sizing is a hard constraint. Classify the region as wide, tall, square, or compact from `WIDTH / HEIGHT`, then choose a composition that belongs in that shape. Do not force a landscape diagram into a tall or narrow block.

- Outer inset: normally `max(8, min(WIDTH, HEIGHT) * 0.04)`; reduce it only for extremely compact areas.
- Content gap: normally `max(6, min(WIDTH, HEIGHT) * 0.025)`.
- Body text target: 12 px at the supplied dimensions; use 10 px only for unavoidable secondary metadata in compact areas.
- Keep labels short. If essential text does not fit, restructure the diagram, wrap it with explicit `<tspan>` lines, or reduce detail. Never allow collisions, clipping, or overflow.
- Derive coordinates from the actual dimensions. Do not hard-code a fixed 1200x800 composition and scale it blindly.
- Use `textLength` or `lengthAdjust` sparingly; fixing the layout is preferable to squashing text.

If dimensions are absent, infer a reasonable aspect ratio and use a fallback viewBox, but treat that as a fallback only. Harness dimensions always take precedence.

## Content-to-composition procedure

Before writing markup, perform this internal sequence:

1. Extract entities, groups, values, sequence, hierarchy, comparison axes, inputs, outputs, and the one or two most important takeaways.
2. Select the lowest-complexity visual grammar that preserves the relationships:
   - sequence or pipeline: left-to-right for wide regions, top-to-bottom for tall regions;
   - hierarchy or taxonomy: tree, layered stack, or nested grouping;
   - time or stages: timeline or stepped progression;
   - quantities: bars, dots, lines, or a compact table with a visible scale;
   - alternatives: aligned comparison columns or lanes;
   - system or method: grouped architecture with directional connectors;
   - abstract concept: a small number of labeled shapes with explicit relationships.
3. Rank the content. Give the primary message the strongest visual weight, supporting relationships medium weight, and incidental detail the weakest weight. Omit decoration that competes with meaning.
4. Sketch a layout grid inside the exact bounds. Reserve space for headings, labels, connectors, legends, and breathing room before placing main objects.
5. Choose a semantic visual language. Shapes must communicate role consistently: inputs, processing, decisions, outputs, warnings, and outcomes should not look interchangeable.
6. Build background, groups, connectors, objects, labels, and annotations in that order. Keep connectors behind nodes so lines do not cross through labels.
7. Apply the validation checklist and revise geometry before returning the SVG.

## Visual quality rules

- Start with a restrained neutral or transparent background unless requested otherwise. Use one accent color for the main emphasis and a small semantic palette for secondary roles.
- Ensure strong text/background contrast. Do not encode meaning by color alone; pair color with position, shape, labels, or line style.
- Establish hierarchy through size, weight, spacing, and grouping. Avoid a collection of equally loud cards.
- Use consistent corner radius, stroke width, arrowhead style, font stack, and spacing. Rounded rectangles are for meaningful containers or nodes, not decoration.
- Prefer direct labels over legends in compact regions. Use legends only when they materially reduce repetition.
- Use whitespace to separate groups, not oversized empty margins. Fill the region with a balanced composition without meaningless ornaments.
- Use subtle borders, dividers, and shadows only when they improve grouping or depth. Avoid heavy shadows, gratuitous gradients, glow effects, and noise.
- Use `font-family="Arial, Helvetica, sans-serif"` or another system fallback stack. Never depend on a custom font.
- Add `<defs>` only for reusable markers, gradients, or filters that are actually used. Prefer solid fills and simple strokes for portability.
- For charts, show a baseline or scale, label units, and preserve honest proportions. Do not invent data; label illustrative values as illustrative.

## Text and localization

- Preserve supplied terminology, capitalization, numbers, and units. Do not rewrite technical claims into unsupported claims.
- Use visible text only when it contributes to understanding. The visual should remain understandable without a paragraph of explanation.
- Estimate text width before placing it. Center text only when the container has enough width; otherwise use left alignment.
- For multiline labels, use explicit `<tspan x="..." dy="...">` elements and calculate the full block height before centering it.
- Do not rotate body text. Rotate an axis label only when geometry makes a horizontal label clearly worse.
- If content is too dense, preserve the highest-value relationships, shorten labels without changing meaning, and group detail. Never shrink all text until unreadable.

## Technical constraints

- Use valid XML-style attribute quoting and close every element.
- Use IDs that are unique within the document. Define referenced markers and filters before use.
- Avoid `foreignObject`, embedded HTML, external CSS, JavaScript, base64 assets, and browser-specific features unless explicitly required.
- Keep path data and geometry simple enough to render reliably. Avoid excessive element counts.
- Use `stroke-linecap="round"` and `stroke-linejoin="round"` consistently for diagrams.
- Use `vector-effect="non-scaling-stroke"` only when constant stroke weight under harness resizing is explicitly desired.

## Final validation checklist

Before responding, verify:

- Root dimensions exactly match the harness dimensions.
- `viewBox` begins at zero and has the correct width and height.
- No element, stroke, marker, label, or shadow is clipped at an edge.
- The composition matches the aspect ratio and uses the available region intentionally.
- The primary message is identifiable at a glance.
- Relationships and directionality are unambiguous without relying only on color.
- Text is readable at the actual supplied size, with no collisions, truncation, or accidental wrapping.
- Supplied facts are represented accurately, with no invented values or claims.
- Colors have sufficient contrast and remain coherent across roles.
- The document is self-contained, valid SVG, and contains no output outside the root `<svg>` element.

When any check fails, change the layout or simplify the content before returning the SVG. The final response must still contain only the corrected SVG document.
