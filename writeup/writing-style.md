# Academic Writing Style Guide — Oratile Nailana

Extracted from the MSc proposal (*Diffusion-Based Refinement of Textual
Conditioning*). This documents the observable style patterns so subsequent
dissertation chapters read as one continuous voice.

## Voice and stance

- **Impersonal, third-person throughout.** No "I". First-person plural ("we")
  is used sparingly and only for the study's own methodological choices
  ("we consider", "we evaluate", "we compare"). Never "you".
- **Measured and hedged, never assertive.** Claims are framed as
  interpretation, not fact: "may be interpreted as", "suggests that",
  "this indicates", "from this perspective", "can be understood as". The
  proposal almost never says "X is"; it says "X can be viewed as" or
  "X emerges as".
- **Framing-forward.** Ideas are repeatedly re-framed through a lens:
  "From a signal processing perspective…", "From this perspective…",
  "In this view…", "Under this decomposition…". This is a signature move —
  each subsection often opens or closes by recasting the prior point.

## Sentence and paragraph construction

- **Long, subordinate-clause-heavy sentences.** Typical sentence carries two
  or three clauses joined by "where", "while", "however", "although", "as a
  result", "consequently". Example rhythm: *"While X improves Y, it introduces
  Z, which in turn leads to W."*
- **Paragraphs are 3–6 sentences**, each making one point then transitioning.
  Paragraphs frequently end with a forward-looking or interpretive sentence
  that motivates the next ("This motivates…", "This suggests a gap in…",
  "This raises the question of whether…").
- **Topic sentences are conceptual, not procedural.** A paragraph opens by
  naming the concept ("Variational inference introduces a tractable surrogate
  distribution…"), not by describing what the paragraph will do.
- **Transitions are explicit connectives**: "However", "In contrast",
  "Consequently", "More recently", "Complementary to this", "Importantly",
  "Furthermore". Sentences rarely start cold.

## Technical exposition

- **Define then formalise.** A concept is stated in prose, then given a
  numbered equation, then the equation's terms are unpacked in an itemised
  list ("where: \n - X = …\n - Y = …"). This define→equation→itemise pattern
  is used consistently.
- **Equations are numbered display equations** (`\begin{equation}`), integrated
  grammatically into the sentence ("is defined as:", "can be expressed as:",
  "yields:").
- **Every symbol is glossed.** After an equation, variables are defined in a
  bulleted `itemize` with `\(x\)` inline math and an em-dash or "=" gloss.
- **Interpretation follows formalism.** After presenting the maths, a sentence
  explains what it *means* conceptually ("This objective can be interpreted as
  learning a denoising direction at each noise level…", "This formulation
  reframes inference as an optimisation problem…").

## Citation habits

- **Dense, clustered citations** at the end of claims, often 2–4 keys:
  `\citep{a, b, c}`. Nearly every non-trivial sentence in Background carries a
  citation cluster.
- **`\citep`** for parenthetical, **`\cite`/`\citet`** for in-text author
  ("`\cite{hertz2022prompt} demonstrate that…`").
- Citations support the *claim*, placed at sentence end before the full stop.

## Punctuation

- **Do not use em dashes** (`---` in LaTeX, or `--`) for parenthetical asides
  or emphasis. This is a firm preference. Rewrite around them:
  - For a parenthetical aside, use commas, parentheses, or split into two
    sentences: "the encoder, a property of the backbone, is held fixed", or
    "the encoder is held fixed (a property of the backbone)".
  - For an appositive or definition, use a comma or a colon, or restructure
    with "where": "one expert per constraint, corresponding to a role-filler
    binding".
  - For a dramatic pause or elaboration, start a new sentence.
- Colons are fine to introduce a list, an equation ("is defined as:"), or an
  elaboration. Semicolons are fine to join two closely related independent
  clauses.

## Vocabulary and phrasing

- **Preferred connective/framing phrases** (reuse these):
  "from this perspective", "in this view", "can be interpreted as",
  "this suggests", "as a result", "consequently", "more recently",
  "complementary to this", "importantly", "this motivates",
  "this raises the question of whether", "emerges as", "reframes … as".
- **Preferred technical verbs**: "introduces", "formalises", "operationalises",
  "amplifies", "preserves", "entangles", "compresses", "reframes", "grounds".
- **Recurring conceptual vocabulary** (the proposal's thematic lexicon):
  "conditioning signal", "role-filler associations/bindings", "semantic
  structure", "compositional fidelity", "representational bottleneck",
  "conditioning-induced distortion", "compressed observation", "latent
  semantic structure", "denoising direction", "amplified under guidance".
- **British/SA spelling**: "modelling", "optimisation", "parameterised",
  "generalise", "behaviour", "colour", "fulfilment". (Match this exactly.)
- Em-dashes rendered as `---` in LaTeX; used for parenthetical asides.
  \emph{(Superseded: do not use em dashes. See the Punctuation section.)}

## Structural conventions

- **Chapters open with a short orienting paragraph** stating what the chapter
  covers and how it connects to the prior one.
- **Bold inline lead-ins** for defined terms on first use: `\textbf{attribute
  misbinding}`, `\textbf{generative modelling}`.
- **Sections summarise upward**: Background ends with "Summary and Research
  Gap"; methodology ends with a "Summary". Each ties findings back to the
  research questions.
- **The conditioning-distortion framework** (the δ term, the
  $\|\delta(c_{ref})\| < \|\delta(c_{raw})\|$ hypothesis) is the theoretical
  spine and is referenced across chapters as the unifying formalism.

## Things to avoid (not in the proposal's voice)

- No contractions.
- **No em dashes.** See the Punctuation section; rewrite around them with
  commas, parentheses, colons, or a new sentence.
- No bullet-point *arguments* (bullets are only for definitions, options, or
  enumerations — never for making a claim).
- No short punchy sentences for emphasis; the register stays even and formal.
- No direct address, rhetorical questions (except the deliberate "This raises
  the question of whether…" framing), or informal intensifiers ("very",
  "really", "a lot").
- No undefined symbols or unglossed equations.
