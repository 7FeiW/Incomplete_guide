# Academic Writing with Codex and Claude

Writing a paper from a Python research project means connecting code, experiment
records, figures, and literature to a clear scientific argument. Codex and Claude
can help organize and revise that material. The author still needs to decide
what the evidence supports and verify the manuscript before sharing it.

This chapter focuses on Codex and Claude Code working with manuscript files in a
research repository. The prompts also work as starting points in a chat interface
when you supply the relevant text. File access and available tools depend on the
client and its permissions; ask the assistant to identify what it actually read.

Use [chapter 14](14_Programming_with_LLM_Agents.md) for the agent working loop and
[chapter 15](15_Agentic_Workflow.md) for shared knowledge, rules, and plans. Here,
the task is turning verified research material into a reviewable manuscript.

## Table of Contents

1. [Choose a Bounded Writing Task](#choose-a-bounded-writing-task)
2. [Prepare the Manuscript Context](#prepare-the-manuscript-context)
3. [Set Up Manuscript Agent Rules](#set-up-manuscript-agent-rules)
4. [Use Focused Writing Skills](#use-focused-writing-skills)
5. [Review with Separate Roles](#review-with-separate-roles)
6. [Preserve Author Style](#preserve-author-style)
7. [Connect Claims to Evidence](#connect-claims-to-evidence)
8. [Draft and Revise in Passes](#draft-and-revise-in-passes)
9. [Review Figures and Reviewer Responses](#review-figures-and-reviewer-responses)
10. [Check Before Sharing](#check-before-sharing)

## Choose a Bounded Writing Task

Start with a task whose output you can inspect. Asking for a complete paper from
a vague research idea leaves too many opportunities for invented methods,
results, and citations.

| Task | Supply | Review for |
| --- | --- | --- |
| Outline a section | Research question, supported findings, audience | Argument order and missing evidence |
| Draft methods | Verified protocol, code, configurations, run records | What was actually done versus what code permits |
| Draft results | Checked summaries, figures, metric definitions | Numbers, units, comparisons, and uncertainty |
| Revise language | Existing prose and terminology | Changes to meaning, scope, and confidence |
| Compare literature | Verified source notes and relevant passages | Whether each source supports the comparison |
| Prepare a reviewer response | Reviewer comment and completed changes | Whether the response accurately describes the revision |

Treat an agent's critique as a list of questions to investigate. A confident
assessment of novelty or methodological correctness needs your own literature
and scientific review. If you compare Codex and Claude for your workflow, give
them the same bounded task and judge factual accuracy, meaning preservation,
review effort, and the usability of the resulting edits.

## Prepare the Manuscript Context

Keep manuscript context close to the research record. The following is an
optional layout inside a Python project, not a required structure for every paper:

```text
sample-project/
├── src/
├── configs/
├── docs/
│   ├── findings/
│   └── plans/
└── manuscript/
    ├── README.md
    ├── outline.md
    ├── claims.md
    ├── sources.md
    ├── writing-rules.md
    ├── sections/
    │   ├── methods.md
    │   └── results.md
    └── figures/
```

`manuscript/README.md` should identify the manuscript entry point, target
audience, current stage, and any existing build or export procedure. The section
files can instead be LaTeX or another format supported by your project. Keep
large outputs and restricted data in their established storage locations.

Link to canonical findings and experiment records rather than copying them into
multiple manuscript notes. An experiment record should identify the code
revision, environment, configuration, seeds where applicable, input provenance,
and output location, as described in
[chapter 15](15_Agentic_Workflow.md#experiment-records).

For repository instructions, Codex uses `AGENTS.md` and Claude Code supports
`CLAUDE.md`. Their loading rules differ; see the
[official Codex instruction guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [Claude Code project-memory documentation](https://code.claude.com/docs/en/memory).
Merge pointers into existing instructions and keep shared writing constraints in
one file. Do not assume a Markdown link automatically loads its target.

For example, put these constraints in `manuscript/writing-rules.md` and explicitly
ask the agent to read that file before editing:

```markdown
# Manuscript writing rules

- Read manuscript/README.md and the evidence named in the task before editing.
- Preserve scientific meaning, terminology, units, numbers, and citation keys.
- Distinguish observations, interpretations, and untested hypotheses.
- Do not invent references, results, procedures, or claims of novelty.
- Flag missing evidence with an HTML TODO comment in Markdown drafts.
- Edit only the files named in the task; report any proposed scientific changes.
- Report sources actually read and checks actually performed.
```

These are instructions for the example project, not hard access controls. For
LaTeX drafts, use `% TODO:` comments instead of HTML comments. Review unresolved
notes before export because comments may disappear from the rendered manuscript.

## Set Up Manuscript Agent Rules

Put recurring constraints in agent instructions so each writing task starts with
the same expectations. Keep task-specific steps in skills and scientific facts
in the research record. The instruction file should tell the agent what to read
and what to preserve; it should not become a second copy of the manuscript.

For the sample layout above, merge this entry point into the project's root
`AGENTS.md` for Codex or `CLAUDE.md` for Claude Code. All paths in this template
are relative to the project root. Create the referenced manuscript files first.

```markdown
## Academic writing

For manuscript tasks:

- Read manuscript/README.md and manuscript/writing-rules.md before working.
- Read manuscript/claims.md, manuscript/sources.md, and the evidence relevant
  to the requested section. Identify inaccessible or missing sources.
- Read the task plan named in the request when one exists.
- Follow the requested scope and distinguish drafting, language editing,
  scientific review, and new analysis.
- Return a reviewable diff for edits or a location-specific report for reviews.
- Keep reviewers read-only and have them review the same recorded draft version.
- Consolidate review findings before editing; report conflicting recommendations.
- After style edits, check the diff for altered claims, uncertainty, and citations.
```

Extend the shared `writing-rules.md` example with constraints appropriate to your
manuscript. A useful starting set is:

| Rule | What the agent should do |
| --- | --- |
| Evidence before assertions | Link consequential claims to source passages or recorded results; flag unsupported claims |
| Preserve scientific meaning | Keep negation, causal language, scope, uncertainty, numbers, units, and equations intact during language edits |
| Citation integrity | Preserve existing citation keys; verify new metadata and supporting passages before proposing a reference |
| Consistent terminology | Follow agreed definitions and abbreviations; report conflicting usage before choosing a new term |
| Preserve author voice | Use direct prose and supplied style examples; avoid adding promotional language or unsupported novelty claims |
| Separate writing from analysis | Report requests for new calculations or experiments as separate work unless the task authorizes them |
| Protect manuscript structure | Preserve labels, cross-references, figure paths, bibliography structure, and the existing export procedure |
| Honest completion records | Name checks actually performed, remaining TODOs, and unresolved scientific decisions |

These rules can live in the shared writing file so both agents use the same
constraints. For directory-specific instructions and permission controls, use
the mechanisms described in [chapter 15](15_Agentic_Workflow.md#agent-guidance).

Check the setup in a fresh session before delegating a revision:

```text
Read the repository instructions and manuscript/writing-rules.md.
For a language edit to manuscript/sections/results.md, summarize the applicable
constraints, evidence files, permitted edits, and expected checks. Identify
missing files or conflicting instructions. Do not edit files.
```

Compare the response with the actual files. Correct stale or conflicting rules
before proceeding.

## Use Focused Writing Skills

A skill packages a repeatable procedure with its inputs, steps, outputs, and
stopping conditions. Use one when you repeat a task often enough to benefit from
a consistent review process. A skill does not provide access to papers or certify
scientific correctness.

The following names are proposed project skills you can create; they are not
claims that these skills are installed or published:

| Proposed skill | Inputs | Output and boundary |
| --- | --- | --- |
| `citation-audit` | Section, source notes, bibliography, original passages | Report of supported, unsupported, and unverified claims; no automatic reference replacement |
| `evidence-to-section` | Accepted outline and verified claim records | One section draft with evidence pointers and TODOs; no invented findings |
| `scientific-consistency` | Manuscript, figures, metric definitions, result summaries | Located discrepancies in numbers, terms, scope, and uncertainty; scientific decisions left open |
| `logic-review` | Research question, outline, section, claim records | Report of missing premises, contradictions, unsupported inferences, and alternative explanations |
| `writing-review` | Section, audience, terminology, writing rules | Report of unclear sentences, weak paragraph flow, repetition, and undefined terms |
| `language-edit` | Named section, writing rules, permitted style sample | Small prose diff plus meaning-sensitive edits flagged for review |
| `author-style` | Reviewed section, agreed style profile, permitted author samples | Natural prose in the author's voice, with scientific meaning preserved and sensitive edits flagged |
| `reviewer-response` | Reviewer comment, revision record, current diff | Response tied to completed changes; unfinished experiments identified as pending |

Start with `citation-audit` and `language-edit` if those are your recurring
bottlenecks. Keep their procedures in shared Markdown files and use thin
agent-specific wrappers, following
[chapter 15's reusable workflow pattern](15_Agentic_Workflow.md#reusable-workflows).

For example, save a citation-audit runbook as
`manuscript/workflows/citation-audit.md` in the sample project. Its procedure can
be:

1. Read the writing rules and the manuscript section named in the request.
2. List externally sourced claims and their existing citation keys.
3. Check bibliographic metadata against the original source or publisher record.
4. Compare each claim with the relevant original passage and its limitations.
5. Report the manuscript location, source location, assessment, and next action.
6. Mark unavailable sources unverified and unsupported claims unsupported.
   Do not edit the manuscript or bibliography during this audit.

Wrap that runbook in `.agents/skills/citation-audit/SKILL.md` for Codex or
`.claude/skills/citation-audit/SKILL.md` for Claude Code:

```markdown
---
name: citation-audit
description: Audit manuscript citations and source support when requested.
---

# Audit manuscript citations

Resolve paths from the project root. Read manuscript/writing-rules.md and
manuscript/workflows/citation-audit.md, then follow the runbook for the section
named in the request. If no section is named, request the target before auditing.
Return a read-only report with precise manuscript and source locations.
Do not invent source metadata or treat unavailable sources as verified.
```

After creating the runbook and wrapper, invoke `$citation-audit` in Codex or
`/citation-audit` in Claude Code with the target section. Verify that the agent
reads the shared procedure. See the
[official Codex skills documentation](https://learn.chatgpt.com/docs/build-skills)
and [Claude Code skills documentation](https://code.claude.com/docs/en/skills)
for current discovery and invocation behavior.

When adopting an existing writing skill, inspect its instructions and any
scripts before use. Check that it preserves scientific uncertainty, supports
your manuscript format, and reports missing sources. Add project-specific
evidence rules where needed. Language cleanup or readability scoring should be
followed by the same meaning and citation checks as any other edit.

## Review with Separate Roles

Use several focused skills when one general review produces vague feedback or
mixes scientific questions with sentence edits. A **skill** defines a procedure;
an **agent** performs it with a particular context and tool access. One agent can
run these skills in separate passes. Separate reviewer sessions or agents may
help keep the reviews focused when the manuscript is substantial. More agents
also add coordination and review effort, and their findings may share the same
blind spots.

Use the following roles as an optional review team. They refer to the proposed
skills above and do not require a particular product's multi-agent feature:

| Reviewer role | Skill or skills | Questions to answer |
| --- | --- | --- |
| Argument reviewer | `logic-review` | Does the argument answer the research question? Do conclusions follow from the stated premises? Are alternatives and limitations addressed? |
| Evidence reviewer | `citation-audit`, `scientific-consistency` | Do sources and results support the claims? Do values, definitions, and uncertainty agree across the manuscript? |
| Writing reviewer | `writing-review` | Can the intended reader follow each sentence and paragraph? Are transitions meaningful and terms defined? |
| Style reviewer | Review procedure from `author-style` | Does the prose follow the agreed author voice? Where do generic phrasing, inflated language, or mechanical repetition obscure meaning? |

Keep review and revision separate. Reviewers should report findings without
editing shared manuscript files. Give each reviewer the same recorded draft
version, relevant context, shared rules, and its own review brief. If running
reviews concurrently, use separate reports and keep the draft unchanged until
they finish. For a small section, sequential passes may be easier to manage.

A useful process is:

1. Record the draft revision and any uncommitted changes included in the review.
2. Run argument and evidence reviews; run writing and style reviews on the same
   draft if feedback on presentation is also useful.
3. Consolidate duplicate findings and separate scientific issues from optional
   prose suggestions. Preserve disagreements and unavailable-evidence notes.
4. Have the author resolve scientific questions and choose the revisions.
5. Have one editor apply the selected changes, then perform the style pass.
6. Recheck changed claims and their support, plus meaning-sensitive language
   edits. Record remaining issues and checks in the task plan.

For example, give an argument reviewer this brief:

```text
Review manuscript/sections/results.md using the logic-review procedure.
Read the research question, accepted outline, claim records, and shared rules.
Check premises, inference steps, contradictions, causal claims, generalization,
and alternative explanations. Do not edit files or rewrite sentences for style.
For each finding, give its location, the relevant text, the reasoning problem,
its consequence, supporting evidence or missing context, and a suggested next
action. Distinguish a demonstrated issue from a question needing author review.
Report what you could not assess and identify the draft version you reviewed.
```

Use the same finding format for the other reviewers, changing the review scope.
Store reports in an optional `manuscript/reviews/` directory or the task record.
Use section names and short text excerpts when line numbers may change. Avoid
counting reviewer agreement as verification; resolve findings against the source
material and scientific reasoning. "No findings" means only that the reviewer
reported none within the stated scope and accessible context.

## Preserve Author Style

An `author-style` skill can make a reviewed draft read more naturally by removing
stock phrases, unnecessary formality, repetitive transitions, and vague wording.
Define "humanizing" as improving readability and preserving the author's voice.
Do not introduce anecdotes, intentional errors, unsupported opinions, or stronger
scientific claims to make text seem personal.

Keep an agreed style profile in `manuscript/style.md`. If you provide examples,
use writing you are permitted to share and ask the agent to propose preferences
for you to review. It should learn features of the style without copying sample
sentences or importing their scientific content. A starting profile is:

```markdown
# Author style

- Audience: research programmers with varying software-engineering experience.
- Use direct, concrete prose and explain specialized terms on first use.
- Keep paragraphs focused and vary sentence length when it improves flow.
- Prefer informative transitions to stock opening and closing phrases.
- Remove inflated adjectives and vague claims of importance or novelty.
- Preserve necessary technical vocabulary, scientific hedging, and precision.
- Follow the manuscript's agreed spelling and first-person conventions.
- Use supplied author samples as style references, not as factual sources.
```

For a paper, replace the audience and conventions with those agreed by its
authors. Keep journal formatting requirements alongside the profile and flag
conflicts instead of silently choosing between them.

The style skill should support a read-only review and an explicitly requested
edit. Its runbook can follow these steps:

1. Read the named section, writing rules, style profile, and permitted samples.
2. Identify specific phrasing that conflicts with the profile; preserve wording
   whose technical meaning requires it.
3. For review, return located suggestions. For editing, apply small changes only
   within the requested section.
4. Compare the before and after text for numbers, units, citation keys, equations,
   negation, causal language, scope, and uncertainty.
5. Flag any meaning-sensitive change and return the diff with unresolved issues.

For example, after the scientific revisions are accepted:

```text
Apply the author-style procedure to manuscript/sections/results.md.
Read manuscript/style.md and manuscript/writing-rules.md first. Improve natural
flow and remove generic or inflated phrasing while preserving technical meaning,
evidence, uncertainty, and citations. Do not add content from style samples.
Return a small diff and flag any edit that may affect scientific interpretation.
```

Evaluate the pass by readability, fit to the author voice, and meaning
preservation. An AI-detector score does not establish scientific accuracy or
authorship. Keep any disclosure record tied to the assistance actually performed.

## Connect Claims to Evidence

A bibliography entry tells you where a source is identified. It does not prove
that the source supports a particular sentence. Keep a compact claim record
while drafting, especially for quantitative findings and literature comparisons.

For each consequential claim, record:

- the proposed statement and its manuscript location;
- the supporting source, run record, or checked result artifact;
- the relevant page, section, table, figure, or result field;
- the conditions and limitations under which the claim holds; and
- its review status, including unresolved questions.

For a classifier paper, a claim record might begin with this template. Replace
the placeholders with verified project evidence; this is not an example result:

```markdown
## C1: Model comparison on the held-out split

- Proposed claim: <bounded comparison supported by the analysis>.
- Manuscript location: sections/results.md, model-comparison paragraph.
- Evidence: <run-record path and exact result artifact or field>.
- Metric definition: <metric, aggregation, units, and evaluation population>.
- Conditions: <split, seeds, preprocessing, and baseline configuration>.
- Limitations: <uncertainty and scope of generalization>.
- Review status: pending author verification.
```

For literature, use `sources.md` to record verified bibliographic metadata,
citation keys, precise source locations, and your notes on relevance and limits.
Read the original source before accepting a claim. A search snippet or abstract
may establish a lead but may omit conditions needed for your sentence. Check PDF
extraction against the original for equations, tables, symbols, and page numbers.

Ask the agent to separate discovery from verification:

```text
Audit the literature claims in manuscript/sections/methods.md against
manuscript/sources.md and the source passages supplied for this task.
For each claim, report its location, citation key, supporting passage location,
and whether the passage supports its scope. Mark sources you cannot access as
unverified. Do not create bibliography entries or edit the draft.
```

If external search is available, ask for candidate sources and verify them before
adding them. Do not let the assistant fill a citation gap from memory. In a
Markdown working draft, a useful unresolved note is:

```markdown
<!-- TODO: verify a primary source for this preprocessing choice and its limits. -->
```

## Draft and Revise in Passes

Separate argument development from language polishing so you can see what each
pass changes. The sequence below is a starting point; skip passes that your
manuscript does not need.

### 1. Outline from Supported Findings

Supply the research question, intended contribution, audience, and evidence
record. Ask for an outline with the evidence required by each paragraph. Resolve
missing support before asking for connected prose.

```text
Read manuscript/outline.md, manuscript/claims.md, and the findings linked there.
Propose a results-section outline. For each paragraph, name its purpose and
supporting claim IDs. List gaps separately. Do not add findings or edit files.
```

### 2. Draft One Section

For methods, provide the actual run configurations and records as well as the
relevant code. A configurable option in today's code does not establish what an
earlier experiment used. For results, provide checked summaries and metric
definitions; leave new analysis as a separate task with its own validation.

```text
Draft manuscript/sections/results.md using the accepted outline and only the
verified entries in manuscript/claims.md. Read the linked evidence first.
Separate observed results from interpretation. Preserve values, units, and
uncertainty descriptions. Add TODO comments for missing support. Do not change
analysis code, result artifacts, or other manuscript files. Leave a reviewable
diff and list the evidence you used.
```

### 3. Review the Scientific Argument

Read the draft against the evidence yourself. You can also request a read-only
critique for unsupported generalization, inconsistent definitions, alternative
explanations, and missing limitations. Resolve proposed scientific changes before
allowing the next edit.

Watch for language that strengthens a claim. Replacing "was associated with" by
"caused", or "on this dataset" by "in general", changes the scientific meaning.
Likewise, describing an observed difference as statistically significant requires
support from the relevant analysis.

### 4. Edit Language While Preserving Meaning

```text
Edit manuscript/sections/results.md for grammar, clarity, and repetition.
Follow manuscript/writing-rules.md. Preserve numbers, citation keys, technical
terms, comparisons, scope, and confidence. Keep the author's direct style.
If a sentence needs a scientific decision, flag it separately rather than
resolving it. Summarize any edit that may affect meaning.
```

Compare the revision with the original sentence by sentence. Use your own prose
as a style reference only when you are permitted to share it. Prefer concrete
style instructions over a vague request to "make it more academic", which can
encourage inflated language. For translation, check technical terms, negation,
and uncertainty against the original as well.

### 5. Write the Abstract from the Reviewed Manuscript

Once the main sections are reviewed, ask for an abstract under the required word
limit using only claims present in the manuscript. Check that the abstract and
conclusion preserve its limitations and introduce no new results.

## Review Figures and Reviewer Responses

For a caption, supply the figure, plotting inputs, and definitions of axes,
units, groups, sample counts, aggregation, and error bars. Ask the agent to draft
a caption from those inputs and flag missing definitions. Check the displayed
figure yourself; a caption can sound correct while describing the wrong panel
or an outdated plot.

For reviewer responses, keep each comment connected to a completed change or a
reasoned explanation. Separate proposed work from work actually performed:

```text
Draft a response to the supplied reviewer comment using the completed changes
in the revision record and the current manuscript diff. Identify the relevant
section and describe only verified changes. If the requested experiment has not
been run, list it as pending rather than claiming it is complete. Do not invent
page or line numbers; flag them for checking against the final exported version.
```

Have coauthors review scientific commitments and the tone before sending the
response. An agent's draft is not evidence that an experiment was performed or
that the reviewer concern has been resolved.

## Check Before Sharing

Use this checklist for a section handoff or final manuscript review:

- Check consequential claims against original sources and experiment records.
- Reconcile numbers, units, sample counts, and uncertainty across prose, tables,
  figures, captions, and the abstract.
- Verify references exist, metadata is correct, citation keys resolve, and the
  cited passages support the statements.
- Review the diff for changes to meaning and unintended edits in commands,
  equations, labels, and bibliography entries.
- Resolve TODO notes and verify the rendered manuscript, including references
  and figure placement. Run the project's documented build if one exists.
- Record manuscript and research revisions, evidence used, assistance performed,
  checks completed, and remaining questions in the task record.
- Check the current rules of your institution, collaborators, and target venue
  for permitted AI use and disclosure. Record any required disclosure from the
  work actually performed; this chapter does not establish a universal policy.
- Before supplying material, confirm that its confidentiality, consent, license,
  and applicable service terms permit that use. Local file access alone does not
  establish that processing stays on your machine.

The reviewable output is a manuscript change together with its evidence and
unresolved questions. Human authors remain responsible for accepting the text
and the scientific claims it makes.
