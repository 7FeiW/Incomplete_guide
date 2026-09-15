# Paper Author Style Example

> This is a paper voice profile, not a universal academic style.
> Adapt it to the author, audience, and venue. Scientific accuracy takes precedence.

Use this profile for writing and style tasks. Follow [AGENTS.md](../AGENTS.md)
and the [author-style runbook](workflows/author-style.md) when revising prose.

When drafting or revising paper prose, preserve the author's established academic voice.

## Source basis and scope

This is an adaptable paper voice profile. Calibrate it using author-approved
paper passages that you are permitted to share. Record sample paths and relevant
sections in the target project; no calibration samples are included here.

Use samples to learn how the author motivates problems, explains mechanisms,
compares methods, and interprets results. Coauthored passages represent shared
exposition; do not attribute every sentence to one author. Exclude templates,
bibliographies, quotations, editorial notes, and commented-out alternatives.

Correct grammar and redundancy while preserving explanatory habits. Historical
counts, scientific claims, placeholders, and promotional wording in samples are
not verified evidence or style requirements. Proposed work does not establish
that an experiment was completed. Follow the agreed venue requirements and
paper conventions when they differ from a sample's formatting.

## Writing style

- Use an explanatory, methodical, and problem-driven structure: establish the scientific problem, explain the limitations of existing methods, introduce the alternative, describe how it works, and evaluate its consequences.
- Explain technical ideas operationally: state what enters a system, what happens, and what the system produces.
- Introduce a method or representation by explaining the problem it addresses. Then describe its components and explain why each operation is needed, using purpose clauses such as `to ...` or `so that ...`.
- Distinguish model training from prediction or retrieval at inference time. State what information is available at each stage and what the model must infer.
- Avoid unnecessarily mathematical phrasing. When mathematical terminology or notation does not add precision, describe the concept directly in operational language.
- Make logical relationships explicit. Use contrasts and caveats to clarify the scope and limitations of claims.
- Prefer cautious academic claims over promotional or absolute language.
- Organize complicated ideas into categories, stages, or enumerated limitations when useful.
- Begin paragraphs with the broader point and progressively introduce technical detail.
- Use moderately long sentences, often around 20--25 words, with shorter definitions and longer explanations when needed. Preserve variation in sentence length; do not pad or split sentences merely to meet a word-count target.
- Let paragraphs develop one explanatory purpose through several connected sentences. The samples often sustain four to six sentences, but paragraph length should follow the argument: split when moving from a mechanism to a different comparison, limitation, or evaluation question.
- Repeat a precise technical noun when doing so is clearer than using an ambiguous pronoun, but avoid unnecessary repetition within the same paragraph.
- Use third person by default. Prefer active first-person plural for paper contributions and experimental choices: `we introduce`, `we formulate`, `we evaluate`, `we compare`, and `we find`. Use impersonal constructions for generally accepted processes or properties.
- Use present tense for definitions, method behavior, and what a figure or table shows; use past tense for completed experimental procedures and observations. Use future or conditional language for proposed experiments and possible extensions, according to their actual status.
- Define a concept in words before introducing its mathematical notation. After an equation, explain the role or interpretation of its important terms in prose.
- Describe workflows chronologically with signals such as `First`, `Then`, `Following`, and `Finally`. State both the operation and its purpose or output at each stage.
- Introduce abbreviations and directional task names explicitly, then use them consistently. When two directions are easy to confuse, contrast them directly.
- Define a representation through the meaning of its parts. For example, explain what a node, edge, fingerprint bit, or peak denotes before describing how a model processes it.
- Use parenthetical examples and clarifications, including `such as`, `i.e.`, and `e.g.`, when they make an abstract statement concrete. Do not overload a sentence with multiple parenthetical remarks.
- Refer explicitly to figures, tables, sections, and appendices when they carry supporting evidence. Briefly state what the reader should learn from the referenced item.
- In results prose, establish the evaluated task and comparison, report the outcome with its supporting metric, and interpret its consequence. Lead with the outcome when the setup is already clear; introduce the setup first when the task, candidate set, or metric changes. Prefer concrete comparative language such as `outperformed`, `matched`, `higher`, `lower`, `increased`, or `decreased` over vague claims of improvement.
- When discussing a limitation, explain its practical consequence and, when appropriate, follow it with a possible remedy or future direction.
- End major discussions with a compact synthesis of what the evidence establishes, while preserving uncertainty that the experiment cannot resolve.

## Preferred language patterns

Use the following expressions naturally where appropriate; do not insert them mechanically:

- `However, ...`
- `While ..., ...`
- `Note that ...`
- `For example, ...`
- `For instance, ...`
- `More specifically, ...`
- `That is, ...`
- `In addition, ...`
- `In contrast, ...`
- `For comparison, ...`
- `Therefore, ...`
- `To address this limitation, ...`
- `Finally, ...`
- `This approach ...`
- `Our model ...`
- `Given ...`
- `based on ...`
- `can be viewed as ...`
- `can be used to ...`
- `is constrained to ...`
- `does not require ...`
- `in terms of ...`
- `a set of ...`
- `the performance of ...`
- `At a high level, ...`
- `The goal of ... is to ...`
- `The results are summarized in ...`
- `For more details, refer to ...`
- `In summary, ...`
- `In practice, ...`
- `In theory, ...`
- `Compared with ...`
- `Rather than ..., we ...`

Favor precise paired terminology when it clarifies direction or comparison, such as `spectrum-to-structure` and `structure-to-spectrum`, `input` and `output`, `prediction` and `retrieval`, `known` and `unknown`, `reference-based` and `reference-free`, `parent` and `child`, or `observed` and `predicted`.

A characteristic argumentative pattern is:

> While [existing method] can [advantage], it is limited by [problem]. Therefore, [proposed approach] uses [mechanism] to [goal].

Treat this guidance as a voice profile rather than a rigid template. Preserve technical accuracy, grammatical correctness, and variation in sentence structure.

## Citations and explanation

- Integrate citations with the specific method, dataset, or claim they support.
  Introduce an author narratively when explaining a particular contribution or
  how one method extends another; otherwise, keep the method or scientific
  process as the subject and place the citation beside the supported statement.
- In related work, first identify the shared task or representation, then
  explain the mechanism and distinguishing change of each relevant method.
  Use chronology when it explains an extension, rather than listing papers
  without connecting their contributions.
- Define an evaluation metric in terms of what it measures before interpreting
  its values. Explain whether it measures peak agreement, candidate ranking,
  structural similarity, or another property, and state whether higher or lower
  values indicate better performance when this is not already clear.
- Use a concrete example after the general explanation to show how the
  mechanism works or why a metric behaves as reported. State what the example
  illustrates and keep its evidential scope distinct from aggregate results.
- Retain the paper's citation commands and cross-reference conventions when
  adapting source prose. Do not copy numeric citation text, Zotero document
  links, or manuscript-specific macros as substitutes for resolved references.

## LaTeX and domain conventions

- Follow the venue's formatting for emphasis and technical terms. Define
  abbreviations on first use and preserve existing notation and macros.
- Use full cross-reference names with the project's label-based commands, such as `Section~\ref{sec:methods}` when that label exists. Do not hard-code numbers.
- Capitalize *Chemical Dark Matter* consistently. Use `MS/MS spectra`, not
  informal abbreviations such as `mass spec` or an ambiguous unqualified
  `spectra`.
- Use `reference-free` as the domain-specific adjective in this example and contrast it
  explicitly with `reference-based` where relevant.
- Introduce directional method categories explicitly and use a consistent
  notation, such as `structure-to-spectrum` and `spectrum-to-structure`. Retain
  arrow notation only where the surrounding paper text already uses it.
- Keep model names in plain text after their first introduction unless an
  established LaTeX macro specifies otherwise.
- Prefer `improve` over `enhance`. Avoid em dashes; use commas, semicolons,
  colons, or parentheses.

## Paper argumentative patterns

- Anchor abstract MS/MS concepts with concrete molecules or measured examples
  when the evidence is available, such as caffeine, bisoprolol, tyrosine,
  ocfentanil, 3-Cl-PCP, methylaminomethanol, or adiphenine.
- Use exact quantitative database comparisons only when they are supported by
  a citation or verifiable repository source. Preserve the database version or
  date when a count describes a historical snapshot.
- Use `Recall that ...` or `Recall our definition of ...` to reconnect later
  arguments to previously defined concepts without repeating the definition.
- For several technical obstacles, use a numbered decomposition and connect
  each obstacle to its practical mechanism or consequence.
- Frame the introduction around the problem, existing approaches, limitations,
  and supported contributions. Use prose or a contribution list according to
  the venue and author preferences.
- In results prose, state the controlled comparison and its quantitative
  consequence in the same paragraph. The comparison may open the paragraph or
  provide its closing synthesis, but do not repeat the same result merely to
  occupy both positions. In discussions, state whether the design supports causal isolation, for
  example by identifying controlled factors and remaining confounders. Do not infer causal isolation from a comparison alone.
- When linking spectrum prediction to compound identification, explain the
  intervening retrieval workflow and report its evaluation separately. Keep
  exact identification, structural similarity, and fragment interpretation
  distinct when discussing what an experiment establishes.
- Explain a design choice by comparing the relevant alternatives and their
  practical tradeoff, then state why the selected approach suits the task.
  Identify additional assumptions or information available to a benchmark
  variant before interpreting its performance.
- Use section transitions when they clarify the argument. End the paper with
  supported conclusions and limitations.

## Argument and paragraph patterns

Use topic sentences that make the paragraph's function clear. Common paragraph structures in the author's writing include:

1. Define a problem or concept, provide a concrete example, and explain why it matters.
2. Describe an existing method, identify its benefit, and then state its limitation.
3. Introduce the proposed method, explain its stages in order, and conclude with the capability enabled by those stages.
4. Point the reader to a result, report the principal comparison, and explain the likely reason or implication.
5. State a limitation, identify the evidence for it, and propose a technically plausible remedy.
6. Define an evaluation task, explain the metric and comparison, report the result, and use an individual example to clarify a success or failure when useful.

For comparisons involving several methods, keep a consistent comparison axis. A useful pattern is:

> [Method A] benefits from [advantage]; however, it is limited by [drawback]. In contrast, [Method B] provides [different advantage], but [remaining drawback]. Our approach combines [relevant strengths] by [mechanism].

## Claim strength

- Use `show` or `demonstrate` when the stated conclusion is directly supported by the reported experiment.
- Use `suggest`, `indicate`, `may`, `could`, or `potential` for interpretations, generalization beyond the evaluated data, and future applications.
- Use qualifiers such as `typically`, `often`, `generally`, `approximately`, and `to our knowledge` when their scope is meaningful and defensible.
- Avoid hiding a concrete result behind excessive hedging. Give the measured result first, then qualify its interpretation.
- Distinguish what the model establishes from what remains unknown; formulations such as `Although we cannot establish ..., our experiments indicate ...` fit the author's preferred style.
- Separate an observed result from a proposed explanation. State competing explanations when the experiment cannot distinguish them, and tie a proposed remedy to the specific limitation it would address.

## Patterns to avoid

Unless required by the genre, a quotation, or the underlying evidence, do not:

- Ask rhetorical questions. State the motivating problem or uncertainty directly.
- Address the reader as `you` or give conversational commands such as `consider` or `imagine`.
- Use first-person singular (`I`, `me`, or `my`). Use `we` and `our` for paper contributions and experimental choices, when consistent with the author and venue conventions; use third person or impersonal constructions elsewhere.
- Use contractions such as `don't`, `isn't`, `can't`, or `we've`.
- Use exclamation marks, jokes, personal anecdotes, or casual conversational asides.
- Use emphatic certainty markers such as `obviously`, `clearly`, `certainly`, `definitely`, or `undoubtedly` as substitutes for evidence.
- Claim that an empirical result `proves` a general conclusion. Prefer `shows` or `demonstrates` for direct evidence and `suggests` or `indicates` for interpretations.
- Use emotional or promotional modifiers such as `amazing`, `remarkable`, `remarkably`, `surprising`, `surprisingly`, or `strikingly` unless the comparison makes the description precise and defensible.
- Prefer ornate transitions such as `furthermore`, `nevertheless`, or `nonetheless` when a characteristic transition such as `however`, `while`, `in addition`, or `in contrast` is clearer.
- Begin a concluding section with the stock phrase `In conclusion`. Begin with the principal finding directly or use `In summary` when an explicit synthesis marker is useful.
- Use extended metaphors, decorative historical narratives, or analogies that do not materially clarify a technical idea.
- Use sentence fragments or a sequence of very short, punchy sentences for rhetorical effect. Short sentences remain appropriate for precise definitions and necessary emphasis.
- Dismiss competing methods or characterize them as simply wrong. Describe their relevant advantages, assumptions, and limitations using a consistent comparison axis.
- Leave the basis of a comparison implicit. Identify whether methods differ in accuracy, efficiency, coverage, interpretability, generalization, or another relevant property.
- End a section with an unsupported broad claim. Tie the synthesis to evidence, acknowledge unresolved uncertainty, or identify a concrete direction for future work.
- Reproduce draft artifacts from the source samples, including unfinished sentences, repeated words, spelling errors, unresolved editorial questions, or unsupported superlatives.
