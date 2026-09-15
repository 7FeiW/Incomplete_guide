# Notation Register Example

> This is an illustrative notation register for a computational research paper.
> The symbols below are generic placeholders, not this project's notation.
> Replace every row with the manuscript's actual symbols before use.

One file records every symbol, abbreviation, and naming convention the
manuscript uses. Drafting, mathematics review, and language passes read this
file instead of inferring a definition from a nearby equation, which is how the
same quantity acquires two symbols across sections.

## How to Use This File

- Define a symbol here before it appears in the manuscript, and record where the
  manuscript introduces it.
- Treat this file as the definition of record. Where the draft and this file
  disagree, report the conflict rather than silently changing either one.
- Renaming a symbol is its own revision pass, not part of a language edit. See
  [Changing a Symbol](#changing-a-symbol).
- Keep the entries an author can check. A symbol whose meaning is still
  undecided belongs in [Open Questions](#open-questions), not in the tables
  with an invented definition.

## Symbols

Record the domain and units because they are the checks a review pass can
actually perform. Leave a cell empty only when the concept has no such
attribute, such as a dimensionless ratio or an index set.

| Symbol | Meaning | Type and domain | Units | Introduced in | Notes |
| --- | --- | --- | --- | --- | --- |
| `n` | Number of samples in the evaluation set | Positive integer | — | `sections/methods.tex`, data description | Reserved for the evaluation set; use `m` for training samples |
| `x_i` | Feature vector of sample `i` | Real vector, length `d` | Feature-dependent | `sections/methods.tex`, model definition | Standardized before training; state the scaler in the methods |
| `y_i` | Observed label of sample `i` | Element of the label set | — | `sections/methods.tex`, task definition | Observed value, never the prediction |
| `\hat{y}_i` | Predicted label of sample `i` | Same domain as `y_i` | — | `sections/methods.tex`, model definition | The hat marks an estimate throughout the manuscript |
| `\theta` | Model parameters | Real vector | — | `sections/methods.tex`, training | Distinct from the hyperparameters in the configuration table |
| `\mathcal{D}` | Labeled dataset | Set of pairs `(x_i, y_i)` | — | `sections/methods.tex`, data description | Subscripts `train`, `val`, `test` name the splits |

<!-- TODO: replace the placeholder rows above with the manuscript's symbols and
     record the section where each one is first defined. -->

## Conventions

State the typographic rules once so every section applies them the same way.

- Scalars in italic, vectors in bold lowercase, matrices in bold uppercase, sets
  in calligraphic type.
- A hat marks an estimated or predicted quantity; a bar marks an average over
  the stated population.
- Subscripts index samples, features, or splits; superscripts index iterations
  or model variants. Do not use the same position for both.
- Define an abbreviation at its first use in the abstract and again at its first
  use in the body, then use the short form consistently.
- Write units with the venue's required style and keep the same style in prose,
  tables, figure axes, and captions.

<!-- TODO: confirm these conventions against the target venue's style guide. -->

## Macros

When the manuscript defines LaTeX macros for notation, record them here so an
edit pass uses the macro rather than expanding it by hand.

| Macro | Expands to | Use for |
| --- | --- | --- |
| `\yhat` | `\hat{y}` | Predicted label, so the hat convention changes in one place |
| `\dataset` | `\mathcal{D}` | The labeled dataset |

<!-- TODO: list the macros the manuscript actually defines and where they are
     defined. Remove this table if the project defines none. -->

## Abbreviations

| Abbreviation | Expansion | First use | Notes |
| --- | --- | --- | --- |
| `MAE` | Mean absolute error | `sections/methods.tex`, metrics | Report with the units of the target variable |

<!-- TODO: record the manuscript's abbreviations, including any the target venue
     requires or forbids. -->

## Reserved and Retired Symbols

Keeping this list prevents a symbol from being reused for a second meaning and
records decisions a later reader would otherwise reverse.

- Reserved: symbols this field assigns a fixed meaning that the manuscript
  should not reassign.
- Retired: symbols an earlier draft used, with their replacement, so review
  passes recognize a leftover occurrence as a leftover.

| Symbol | Status | Reason | Replacement |
| --- | --- | --- | --- |
| `p` | Reserved | Reads as a significance value in this field | Use `\pi` for a mixing proportion |
| `N` | Retired | Ambiguous between dataset size and iteration count | `n` for samples, `T` for iterations |

<!-- TODO: record the symbols this manuscript reserves or has retired. -->

## Changing a Symbol

A rename touches prose, equations, tables, figure axes, captions, and
supplementary material, so run it as a separate pass:

1. Update this file first, moving the old symbol to the retired table.
2. Search the manuscript sources for every occurrence, including figure-
   generating code and captions, before editing.
3. Apply the change in one commit whose diff is reviewable on its own.
4. Regenerate affected figures and rebuild the manuscript, then read the
   rendered output; a symbol baked into a figure image does not change with
   the source.

## Open Questions

Record undecided notation here rather than letting a draft settle it silently.

<!-- TODO: list notation decisions awaiting an author or coauthor choice, with
     the sections that depend on each one. -->
