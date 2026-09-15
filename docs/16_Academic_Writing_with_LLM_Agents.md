# Academic Writing with Codex and Claude

Writing a paper from a Python research project means connecting code, experiment
records, figures, and literature to a clear scientific argument. Codex and Claude
can help organize and revise that material. The author still needs to decide
what the evidence supports and verify the manuscript before sharing it.

This section focuses on Codex and Claude Code working with manuscript files in a
research repository. The prompts also work as starting points in a chat interface
when you supply the relevant text. File access and available tools depend on the
client and its permissions; ask the assistant to identify what it actually read.

Published agent systems aim much higher than this. Agent Laboratory reports an
end-to-end pipeline from literature review through a written report
([Schmidgall et al., 2025](https://arxiv.org/pdf/2501.04227)), and a proposed
auto-research framework coordinates agents across eight phases of the research
lifecycle ([Liu et al., 2025](https://arxiv.org/html/2504.18765v1)); both are
preprints describing systems rather than validated results. Read them for what
can be automated. Neither changes who is accountable for a claim, which is why
this section automates only steps whose output an author can check.

Use [guide section 14](14_Programming_with_LLM_Agents.md) for the agent working loop and
[guide section 15](15_Agentic_Workflow.md) for shared knowledge, rules, and plans. Here,
the task is turning verified research material into a reviewable manuscript.

## Table of Contents

1. [Choose a Bounded Writing Task](#choose-a-bounded-writing-task)
2. [Choose a Model for the Writing Task](#choose-a-model-for-the-writing-task)
3. [Prepare the Manuscript Context](#prepare-the-manuscript-context)
4. [Set Up Manuscript Agent Rules](#set-up-manuscript-agent-rules)
5. [Use Focused Writing Skills](#use-focused-writing-skills)
6. [Review with Separate Roles](#review-with-separate-roles)
7. [Preserve Author Style](#preserve-author-style)
8. [Connect Claims to Evidence](#connect-claims-to-evidence)
9. [Draft and Revise in Passes](#draft-and-revise-in-passes)
10. [Review Figures and Reviewer Responses](#review-figures-and-reviewer-responses)
11. [Check Before Sharing](#check-before-sharing)
12. [Further Reading](#further-reading)

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

## Choose a Model for the Writing Task

The largest model is not necessary for every manuscript task. A smaller or
faster model may be sufficient for a fixed-format task such as extracting
citation keys, checking a checklist, or identifying repeated terminology. Use a
more capable model when the work genuinely needs long-context synthesis or
multi-step reasoning, such as reconciling a draft with several supplied evidence
records. Choose a model that supports the required files, context length, and
tools; test it on representative material before using it broadly.

Do not select an "Ultra," "Pro," or maximum-reasoning tier merely because the
task is a paper. Those labels and controls differ between providers, and more
reasoning is not automatically better. Instead, choose the least reasoning
effort that produces an adequately reviewed result for the task.

Both providers expose reasoning depth as a named effort setting, and both advise
starting at the documented default and moving only on evidence. For OpenAI
models, GPT-5.5 defaults to `medium`, described as the balanced starting point;
raise it to `high` or `xhigh` only when an evaluation shows a measurable quality
gain over the added latency and cost. For Claude models, both the available
`effort` levels and the default are model- and surface-dependent, so check the
provider documentation for the model you actually use. Recent models such as
Claude Opus 5 and Claude Sonnet 5 accept `low`, `medium`, `high`, `xhigh`, and
`max`; earlier ones accept a narrower set, and some do not accept the parameter
at all. The API default on models that support the parameter is `high`, whereas
Claude Code defaults to `xhigh` on the models where that level exists. Whatever
the default is for your model, the guidance is the same: step down to `medium`
or `low` for routine work once your own checks show quality holds, and reserve
the top levels for long-horizon work that justifies the token cost. In Claude
Code, `/model` selects the model and the effort selector sets the level, and the
selector shows which levels that model supports. Effort is a behavioral signal
rather than a fixed token budget, so a level that is too high for a bounded task
can produce overthinking, which
[Constrain Overthinking in Manuscript Work](#constrain-overthinking-in-manuscript-work)
covers below.

| Paper task | Starting reasoning effort | Why | When to increase it |
| --- | --- | --- | --- |
| Drafting from an accepted outline; grammar, clarity, and format edits | Low or moderate | The scope and supporting material are already bounded, so the primary need is clear, controlled prose and a reviewable diff. | The edit must reconcile several supplied constraints, such as a journal guide, terminology list, and style profile. |
| Outline critique; argument structure; evidence and citation review | Moderate | The agent must compare claims, sources, limitations, and section goals without silently resolving scientific questions. | The review repeatedly misses cross-section conflicts or cannot trace an inference through the supplied evidence. |
| Reconciling a long manuscript with figures, tables, result records, or multiple supplied sources | High, after a representative comparison | The task may require many dependent checks and long-context synthesis. | Use it only when a lower setting demonstrably misses material discrepancies that the higher setting finds. |
| Formal proof, mathematical derivation, algorithm, or statistical claim review | High only as an additional review pass | These tasks can require multi-step analysis, but a model response is not verification. | Do not increase effort as a substitute for checking the original definitions, calculations, code, and qualified human review. |

For a recurring task, compare the same source packet and review rubric at two
effort levels, or across two candidate models. Measure the errors that matter
for the manuscript—changed numbers, units, or uncertainty; unsupported claims;
missed or altered citations; changed claim scope—alongside editing effort,
latency, and cost. Raise the setting, or route difficult cases to a more capable
model, only when that comparison catches relevant problems reliably enough to
justify the trade-off. A higher setting does not make unverified sources,
incomplete context, or an ambiguous writing request reliable, and every model
output still requires the evidence and meaning checks in this section.

Provider guidance changes. For OpenAI models, consult
[OpenAI's model-selection guide](https://developers.openai.com/api/docs/guides/model-selection)
and [current model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.5).
For Claude models, consult
[Anthropic's model-selection guide](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model),
the [effort parameter reference](https://platform.claude.com/docs/en/build-with-claude/effort),
and [Claude Code model configuration](https://code.claude.com/docs/en/model-config).

### Constrain Overthinking in Manuscript Work

Overthinking is a measured behavior rather than an impression. A benchmark of 53
models on basic arithmetic found reasoning models producing roughly eighteen
times more output tokens than standard models while sometimes scoring lower,
with sharply diminishing returns from larger budgets and contradictions
accumulating in long chains
([Srivastava et al., Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1285/)).
A structural analysis attributes the pattern to over-verification and
over-exploration—re-checking a settled sub-result, or opening alternatives the
task does not need—and proposes defining overthinking by the utility of each
reasoning step rather than by response length
([Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.773/),
[DeepMind summary](https://deepmind.google/research/publications/203490/)).

In manuscript work this is harder to notice than in code, because the output
still reads well. A revision pass hedges a claim the evidence supports, adds a
limitation the data does not require, replaces the author's term with a more
common one, restates a number in a way that changes its uncertainty, or rewrites
a sentence the coauthors already approved. Nothing fails; the meaning moved.
Asking for more thinking does not prevent this. Structure the request instead.

1. **State the scope and the stopping condition.** "Edit only the paragraph
   below for grammar and clarity. Do not restructure the section, add or remove
   citations, or comment on the science."
2. **Separate drafting from checking.** Ask for the draft, then run a separate
   review pass with a named question: "Check this paragraph for numbers, units,
   and uncertainty that disagree with the supplied result records. Report each
   disagreement with its source line."
3. **Require an explicit change criterion.** "Preserve the author's wording
   unless you can quote the sentence and name the specific problem: a grammar
   error, an undefined term, a claim the supplied evidence does not support, or
   a conflict with another section. Do not change wording for preference." A
   review pass without a criterion will produce edits because it was asked for
   edits.
4. **Verify outside the model.** Check numbers against the experiment records
   and figure data, citations against the original sources, and definitions
   against the methods, as in
   [Connect Claims to Evidence](#connect-claims-to-evidence). A second model
   opinion, or the same model agreeing with itself, is not verification.
5. **Match the effort to the task.** Use the table above, and keep bounded
   language edits at a low setting. Each additional reasoning token is another
   opportunity for the model to decide that a correct sentence needs improving.
6. **Keep the manuscript recoverable.** Commit before a revision pass so every
   edit is a reviewable diff, and gate anything you cannot undo: "Show the
   proposed edits as a diff and wait for my confirmation before writing to any
   file. Do not edit the bibliography, equations, labels, or figure files, and
   do not run the build or submit anything."

A request carrying these constraints:

```text
Edit the Results paragraph in manuscript/sections/results.tex for grammar and
clarity only.

Do not restructure the paragraph, change the order of the findings, add
citations, or add interpretation. Preserve every number, unit, and uncertainty
exactly as written, and preserve the terminology in manuscript/style.md.

Keep the author's wording unless you can quote the sentence and name a specific
problem. Preference is not a reason to change a sentence.

Show the result as a diff with a one-line reason for each change. Do not write
to any file until I confirm.
```

The same discipline appears in [Preserve Author Style](#preserve-author-style)
and [Review with Separate Roles](#review-with-separate-roles): a pass that can
change anything for any reason will, and the author then has to reconstruct what
the paragraph originally claimed.

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
    ├── main.tex
    ├── references.bib
    ├── outline.md
    ├── claims.md
    ├── sources.md
    ├── writing-rules.md
    ├── style.md
    ├── workflows/
    ├── sections/
    │   ├── methods.tex
    │   └── results.tex
    └── figures/
```

This example uses LaTeX for the manuscript: `manuscript/main.tex` is its entry
point, and `manuscript/sections/` holds included source files. Keep
`manuscript/README.md` for the target audience, current stage, and documented
build or export procedure. The Markdown files hold supporting planning and
evidence records, and `manuscript/workflows/` holds the shared runbooks described
in [Use Focused Writing Skills](#use-focused-writing-skills). Keep large outputs
and restricted data in their established storage locations.

The directory names are a project choice. Nesting under `manuscript/` suits a
research repository where code and paper sit side by side. A LaTeX-only project
whose sources already live at the repository root may prefer a shorter name such
as `editorial/` for the same runbooks. The rest of this section writes
`manuscript/workflows/` because the sample layout does; substitute the path your
project actually uses.

Link to canonical findings and experiment records rather than copying them into
multiple manuscript notes. An experiment record should identify the code
revision, environment, configuration, seeds where applicable, input provenance,
and output location, as described in
[guide section 15](15_Agentic_Workflow.md#experiment-records).

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
- Never fabricate citations. If a required citation is unknown, add a
  descriptive `% TODO: cite ...` comment that identifies the claim requiring
  support. Do not insert an empty `\cite{}` command or invent authors, titles,
  venues, or publication years.
- Ask when scientific content, the intended argument, or the correct technical
  phrasing cannot be established from the repository or supplied evidence.
- Do not create manuscript figures unless the user explicitly overrides this rule.
  During review-only tasks, report the missing figure without changing the file.
  When an edit requires a figure placeholder, preserve an existing placeholder
  or add a blank figure environment with a descriptive TODO comment, then ask
  what the figure should contain.
- Edit only the files named in the task; report any proposed scientific changes.
- Report sources actually read and checks actually performed.
```

These are instructions for the example project, not hard access controls.
Review unresolved notes before export because LaTeX comments may disappear from
the rendered manuscript.

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
the mechanisms described in [guide section 15](15_Agentic_Workflow.md#agent-guidance).

Check the setup in a fresh session before delegating a revision:

```text
Read the repository instructions and manuscript/writing-rules.md.
For a language edit to manuscript/sections/results.tex, summarize the applicable
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
claims that these skills are installed or published. Seven of them ship as
adaptable examples in this repository and are marked below; the rest are
starting points you would write yourself.

| Proposed skill | Example included | Inputs | Output and boundary |
| --- | --- | --- | --- |
| `evidence-review` | Yes | Section, source notes, bibliography, original passages, result records, metric definitions | Report of supported, unsupported, and unverified claims, plus located discrepancies in numbers, units, terms, scope, and uncertainty; no automatic reference replacement and no new analysis |
| `evidence-to-section` | No | Accepted outline and verified claim records | One section draft with evidence pointers and TODOs; no invented findings |
| `logic-review` | Yes | Research question, outline, section, claim records | Report of missing premises, contradictions, unsupported inferences, and alternative explanations |
| `proof-review` | Yes | Theorem or proposition, proof, definitions, cited prerequisites | Located gaps in inference, assumptions, domains, and edge cases; no invented proof steps |
| `math-review` | Yes | Equations, notation, derivations, and result records | Located inconsistencies in notation, assumptions, transformations, units, or numerical claims; no silent equation changes |
| `algorithm-review` | Yes | Algorithm description, pseudocode, implementation, and analysis records | Located ambiguities and mismatches in inputs, outputs, state, termination, correctness, or complexity; no invented bounds or behavior |
| `writing-review` | Yes | Section, audience, terminology, writing rules | Report of unclear sentences, weak paragraph flow, repetition, and undefined terms |
| `language-edit` | No | Named section, writing rules, permitted style sample | Small prose diff plus meaning-sensitive edits flagged for review |
| `author-style` | Yes | Reviewed section, agreed style profile, permitted author samples | Natural prose in the author's voice, with scientific meaning preserved and sensitive edits flagged |
| `reviewer-response` | No | Reviewer comment, revision record, current diff | Response tied to completed changes; unfinished experiments identified as pending |

### Available Example Skills and Installation

This repository includes seven ready-to-adapt, **project-local** Codex and Claude
Code wrappers in the [agentic writing example](../examples/agentic_writing/).
They are examples, not independently validated assessments of a manuscript.
Each wrapper refers to the example's shared `manuscript/workflows/` runbooks, so
copy and adapt those runbooks before using a wrapper in another project.

| Example skill | Purpose | Codex wrapper | Claude Code wrapper |
| --- | --- | --- | --- |
| `logic-review` | Find missing premises, unsupported inferences, contradictions, and alternative explanations without editing. | [`logic-review`](../examples/agentic_writing/.agents/skills/logic-review/SKILL.md) | [`logic-review`](../examples/agentic_writing/.claude/skills/logic-review/SKILL.md) |
| `proof-review` | Find gaps in a stated mathematical proof without supplying missing proof steps. | [`proof-review`](../examples/agentic_writing/.agents/skills/proof-review/SKILL.md) | [`proof-review`](../examples/agentic_writing/.claude/skills/proof-review/SKILL.md) |
| `math-review` | Report notation, derivation, assumption, unit, and numerical-consistency issues without editing equations. | [`math-review`](../examples/agentic_writing/.agents/skills/math-review/SKILL.md) | [`math-review`](../examples/agentic_writing/.claude/skills/math-review/SKILL.md) |
| `algorithm-review` | Review an algorithm's specification, pseudocode, implementation alignment, and supported claims without editing. | [`algorithm-review`](../examples/agentic_writing/.agents/skills/algorithm-review/SKILL.md) | [`algorithm-review`](../examples/agentic_writing/.claude/skills/algorithm-review/SKILL.md) |
| `evidence-review` | Check whether supplied evidence and citations support the manuscript's claims; leave unavailable evidence unverified. | [`evidence-review`](../examples/agentic_writing/.agents/skills/evidence-review/SKILL.md) | [`evidence-review`](../examples/agentic_writing/.claude/skills/evidence-review/SKILL.md) |
| `writing-review` | Report clarity, structure, terminology, and flow problems without changing the draft. | [`writing-review`](../examples/agentic_writing/.agents/skills/writing-review/SKILL.md) | [`writing-review`](../examples/agentic_writing/.claude/skills/writing-review/SKILL.md) |
| `author-style` | Review or, when explicitly requested, make small style edits while checking for changes in scientific meaning. | [`author-style`](../examples/agentic_writing/.agents/skills/author-style/SKILL.md) | [`author-style`](../examples/agentic_writing/.claude/skills/author-style/SKILL.md) |

To install one of these examples in a manuscript repository, first copy and
adapt the corresponding runbook into your shared-procedure directory, then merge
the example's `AGENTS.md` guidance into the repository's existing instruction
file — `AGENTS.md` for Codex or `CLAUDE.md` for Claude Code. Do not overwrite
existing instructions; the example ships one shared instruction file, not one per
agent. Then place the wrapper directory at `.agents/skills/<skill-name>/` for
Codex or `.claude/skills/<skill-name>/` for Claude Code. Skill directories stay
at the project root because each agent discovers them there; only the runbooks
follow your own layout.

The example uses `manuscript/workflows/`, so a Codex project adopting it as-is
would contain:

```text
<project-root>/
├── AGENTS.md                      # Merge the applicable example guidance.
├── manuscript/
│   └── workflows/
│       ├── review.md
│       └── logic-review.md        # Copy and adapt these shared procedures.
└── .agents/
    └── skills/
        └── logic-review/
            └── SKILL.md           # Copy and adapt the thin wrapper.
```

A LaTeX-only repository, whose manuscript sources already sit at the root, may
prefer a shorter directory for the same files:

```text
<project-root>/
├── AGENTS.md
├── main.tex
├── sections/
├── editorial/
│   ├── review.md
│   └── logic-review.md
└── .agents/
    └── skills/
        └── logic-review/
            └── SKILL.md
```

Either layout works. The wrapper `SKILL.md` is what tells the agent where to
read, so whichever directory you choose, use one location consistently and edit
the copied wrapper's paths to match it.

From the root of this guide, use the following PowerShell commands only when
`<project-root>\.agents\skills\logic-review` does not already exist. They create
the destination skill directory and copy the example; they do not copy the
required shared workflows or merge instruction files for you.

```powershell
$projectRoot = '<project-root>'
$skillTarget = Join-Path $projectRoot '.agents\skills\logic-review'

if (-not (Test-Path -LiteralPath $projectRoot -PathType Container)) {
    throw "Project root does not exist: $projectRoot"
}

if (Test-Path -LiteralPath $skillTarget) {
    throw "Refusing to overwrite existing skill: $skillTarget"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $skillTarget) -Force | Out-Null
Copy-Item -Recurse -Path '.\examples\agentic_writing\.agents\skills\logic-review' -Destination $skillTarget
```

Replace `logic-review` consistently to install one of the other example skills.
Review the copied `SKILL.md` and runbooks for paths, permissions, manuscript
format, and evidence rules before invoking it. In Codex, use `/skills` to check
discovery, then invoke `$logic-review` (or another installed name) with a target
section and draft version. In Claude Code, invoke the corresponding
`/logic-review` command.

For a reusable external skill, prefer an installable plugin. In Codex, use
`$skill-installer` to inspect curated skills or request a specific repository
skill; OpenAI's curated plugin examples are in the
[OpenAI Plugins repository](https://github.com/openai/plugins). In Claude Code,
register a marketplace with `/plugin marketplace add <owner>/<repo>` and install
from it with `/plugin install <name>@<marketplace>`, as described in
[Discover and install plugins](https://code.claude.com/docs/en/discover-plugins).
Treat any external writing skill as untrusted until you have inspected its
instructions, scripts, data access, and license. Do not install a skill merely
because its name suggests it can verify citations or scientific correctness.

### External Skill Sets to Evaluate

There is no central, cross-agent usage registry for academic-writing skills, so
GitHub stars, downloads, or a skill's name do not establish that it is widely
used, maintained, or scientifically reliable. The following are the broadest
public collections located for the proposed skills in this section as of
September 2026. They are candidates to inspect and test with synthetic or
non-confidential material, not endorsements and not a substitute for the local
wrappers above.

| Proposed skill | Closest external candidate | Fit and limitation |
| --- | --- | --- |
| `evidence-review` | [`citation-auditor`](https://github.com/yaotingsun/academic-publishing-skills) from Academic Publishing Skills for the citation half; [`paper-review`](https://github.com/WenyuChiou/academic-writing-skills) and Academic Publishing Skills' `statistical-rigor-helper`, `figure-checker`, and `table-checker` for the consistency half | No single external skill covers both halves. Confirm that `citation-auditor`'s source-access and metadata steps suit the bibliography manager and citation format in use. The consistency candidates separate statistical, figure, and table checks, and none establishes correctness without the underlying records. |
| `evidence-to-section` | [`academic-writing-skills`](https://github.com/WenyuChiou/academic-writing-skills) | Covers evidence-led drafting from an outline and approved evidence. It is a broad workflow rather than a small, independently installable drafting pass. |
| `logic-review` | [`paper-review`](https://github.com/WenyuChiou/academic-writing-skills) | Includes an argument-and-structure pass and is read-only by default. Keep the local `logic-review` wrapper when a smaller, bounded report is preferable. |
| `proof-review` | No verified general-purpose external review skill found | Do not use a proof-writing skill as a proof verifier. Retain the repository's read-only `proof-review` procedure, and have a qualified human verify any proof. |
| `math-review` | No verified general-purpose external review skill found | A mathematics/LaTeX writing skill can format or draft equations but does not validate a derivation. Retain the local read-only review procedure and check against definitions and calculations. |
| `algorithm-review` | No verified general-purpose external review skill found | Keep the local procedure focused on specified inputs, outputs, invariants, termination, and stated complexity; verify conclusions against code and formal analysis. |
| `writing-review` | [`paper-review`](https://github.com/WenyuChiou/academic-writing-skills) or Academic Publishing Skills' `manuscript-copyeditor` | The former is a review pass; the latter is an editing workflow. Choose a read-only review before an editing pass when preserving scientific meaning is important. |
| `language-edit` | Academic Publishing Skills' [`manuscript-copyeditor`](https://github.com/yaotingsun/academic-publishing-skills) | Closest dedicated copyediting workflow. Review diffs for changed negation, uncertainty, units, notation, and citations. |
| `author-style` | [`academic-writing-skills`](https://github.com/WenyuChiou/academic-writing-skills) | Covers terminology, flow, and stock phrasing, but is not a dedicated author-voice matcher. Keep the local style profile and require explicit approval for edits. |
| `reviewer-response` | Academic Publishing Skills' [`revision-responder`](https://github.com/yaotingsun/academic-publishing-skills) | Directly targeted at responding to revisions. Connect every response to a completed, checked change and leave pending work explicit. |

The two broad collections have different packaging. Academic Writing Skills
provides the `academic-writing-skills` and `paper-review` workflows and documents
both a Claude Code marketplace installation and copying the skill folders into a
directory scanned by Codex. Academic Publishing Skills provides separate Claude
Code plugins, including `citation-auditor`, `manuscript-copyeditor`,
`statistical-rigor-helper`, `figure-checker`, `table-checker`,
`revision-responder`, and `proof-checker`; inspect each plugin before installing
only the parts needed. Its `proof-checker` concerns publication proofs, not the
validity of mathematical proofs.

For computational or applied-mathematics manuscript *authoring*, the
[`claude-latex-skill`](https://github.com/hameefy/claude-latex-skill) covers
LaTeX, theorems, proofs, algorithms, and numerical-experiment documents. It is
a drafting and formatting aid, not a review or verification skill; do not map it
to `proof-review`, `math-review`, or `algorithm-review` without adding the
independent checks required by those local procedures.

Two further candidates orchestrate a whole pipeline rather than a single pass:
the ten-stage [`academic-pipeline`](https://claudemarketplaces.com/skills/imbad0202/academic-research-skills/academic-pipeline)
orchestrator skill, and [`ndcorder/research-agent`](https://github.com/ndcorder/research-agent),
a Claude Code toolkit that drafts LaTeX papers through multi-agent
orchestration. Both are small and unvetted, and neither listing site is
affiliated with Anthropic. A pipeline that runs from literature review to a
finished draft also moves the author's checkpoints inside the automation; if you
test one, keep the evidence and review gates in this section as external checks.

Before adopting any candidate, pin or record the reviewed revision, inspect all
`SKILL.md` files and bundled scripts, test it on a copied manuscript or synthetic
example, and keep only the instructions that match the project's evidence,
privacy, and review rules. The broad Academic Writing Skills repository had 48
GitHub stars when this section was checked; that is a discovery signal only, not
evidence of quality or suitability.

Start with `evidence-review` and `language-edit` if those are your recurring
bottlenecks. Keep their procedures in shared Markdown files and use thin
agent-specific wrappers, following
[guide section 15's reusable workflow pattern](15_Agentic_Workflow.md#reusable-workflows).

For a skill with no example in this repository, write the runbook first. For
example, save a language-edit runbook as `manuscript/workflows/language-edit.md`
in the sample project. Its procedure can be:

1. Read the writing rules, the style profile, and the manuscript section named
   in the request.
2. Edit only that section for grammar, clarity, repetition, and paragraph flow.
3. Preserve numbers, units, equations, citation keys, labels, negation, causal
   language, scope, and stated uncertainty.
4. Compare the edited text with the original sentence by sentence and list every
   change that could affect scientific meaning.
5. Leave any sentence that needs a scientific decision unedited and report it.
6. Return the diff, the flagged sentences, and the checks performed.

Wrap that runbook in `.agents/skills/language-edit/SKILL.md` for Codex or
`.claude/skills/language-edit/SKILL.md` for Claude Code:

```markdown
---
name: language-edit
description: Edit manuscript prose for clarity while preserving scientific meaning.
---

# Edit manuscript language

Resolve paths from the project root. Read manuscript/writing-rules.md,
manuscript/style.md, and manuscript/workflows/language-edit.md, then follow the
runbook for the section named in the request. If no section is named, request
the target before editing. Edit only that section and return a small diff.
Flag every edit that may change scientific meaning instead of resolving it.
```

After creating the runbook and wrapper, invoke `$language-edit` in Codex or
`/language-edit` in Claude Code with the target section. Verify that the agent
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

Orchestration frameworks implement that separation of roles directly:
LlamaIndex's agents-as-tools example coordinates separate research, writing, and
review agents beneath one orchestrator
([Multi-Agent Report Generation](https://developers.llamaindex.ai/python/examples/agent/agents_as_tools/)).
Borrow that structure, but keep acceptance of each finding with the author
rather than with a downstream agent.

Use the following roles as an optional review team. They refer to the proposed
skills above and do not require a particular product's multi-agent feature:

| Reviewer role | Skill or skills | Questions to answer |
| --- | --- | --- |
| Argument reviewer | `logic-review` | Does the argument answer the research question? Do conclusions follow from the stated premises? Are alternatives and limitations addressed? |
| Proof reviewer | `proof-review` | Does each proof step follow from its stated assumptions and prerequisites? Are domains, quantifiers, and edge cases handled? |
| Mathematics reviewer | `math-review` | Are notation, derivations, assumptions, units, and numerical claims internally consistent and supported? |
| Algorithm reviewer | `algorithm-review` | Are inputs, outputs, state changes, termination, and correctness or complexity claims specified and consistent with the implementation? |
| Evidence reviewer | `evidence-review` | Do sources and results support the claims? Do values, definitions, and uncertainty agree across the manuscript? |
| Writing reviewer | `writing-review` | Can the intended reader follow each sentence and paragraph? Are transitions meaningful and terms defined? |
| Style reviewer | Review procedure from `author-style` | Does the prose follow the agreed author voice? Where do generic phrasing, inflated language, or mechanical repetition obscure meaning? |

Keep review and revision separate. Reviewers should report findings without
editing shared manuscript files. Give each reviewer the same recorded draft
version, relevant context, shared rules, and its own review brief. If running
reviews concurrently, use separate reports and keep the draft unchanged until
they finish. For a small section, sequential passes may be easier to manage.

A useful process is:

1. Record the draft revision and any uncommitted changes included in the review.
2. Run the argument, proof, mathematics, algorithm, and evidence reviews that
   fit the draft; run writing and style reviews on the same draft if feedback on
   presentation is also useful.
3. Consolidate duplicate findings and separate scientific issues from optional
   prose suggestions. Preserve disagreements and unavailable-evidence notes.
4. Have the author resolve scientific questions and choose the revisions.
5. Have one editor apply the selected changes, then perform the style pass.
6. Recheck changed claims and their support, plus meaning-sensitive language
   edits. Record remaining issues and checks in the task plan.

For example, give an argument reviewer this brief:

```text
Review manuscript/sections/results.tex using the logic-review procedure.
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
Apply the author-style procedure to manuscript/sections/results.tex.
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
that the source supports a particular sentence.

This is measurable, not hypothetical. Across 636 citations generated for 42
topics, 55% of GPT-3.5's citations and 18% of GPT-4's were fabricated outright;
among the citations that did refer to real work, 43% and 24% respectively
carried at least one substantive error, most often a wrong volume, page range,
date, or author name ([Walters and Wilder, *Scientific Reports* 13:14045,
2023](https://doi.org/10.1038/s41598-023-41032-5)). A comparison on
systematic-review references reported hallucination rates of 39.6% for GPT-3.5,
28.6% for GPT-4, and 91.4% for Bard ([Chelli et al., *Journal of Medical
Internet Research*, 2024](https://doi.org/10.2196/53164)). Those rates are tied
to the models and prompts tested and will not transfer to your tool, but the
failure mode does: a plausible citation is not evidence that the work exists,
and a real citation is not evidence that it says what your sentence claims.

Keep a compact claim record while drafting, especially for quantitative findings
and literature comparisons.

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
- Manuscript location: sections/results.tex, model-comparison paragraph.
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
Audit the literature claims in manuscript/sections/methods.tex against
manuscript/sources.md and the source passages supplied for this task.
For each claim, report its location, citation key, supporting passage location,
and whether the passage supports its scope. Mark sources you cannot access as
unverified. Do not create bibliography entries or edit the draft.
```

If external search is available, ask for candidate sources and verify them before
adding them. Do not let the assistant fill a citation gap from memory. In a
LaTeX working draft, a useful unresolved note is:

```latex
% TODO: verify a primary source for this preprocessing choice and its limits.
```

## Draft and Revise in Passes

Separate argument development from language polishing so you can see what each
pass changes. The sequence below is a starting point; skip passes that your
manuscript does not need.

Automated research pipelines also divide the work into ordered stages rather
than one undifferentiated generation step
([Schmidgall et al., 2025](https://arxiv.org/pdf/2501.04227);
[Liu et al., 2025](https://arxiv.org/html/2504.18765v1)), though their stage
boundaries are their own and do not map one-to-one onto the passes below. The
difference that matters here is that each pass ends at an author decision and a
reviewable diff rather than at the next agent call.

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
Draft manuscript/sections/results.tex using the accepted outline and only the
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
Edit manuscript/sections/results.tex for grammar, clarity, and repetition.
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
- Confirm that every edit from a revision pass has a stated reason, and restore
  the author's original wording wherever that reason is only preference.
- Resolve TODO notes and verify the rendered manuscript, including references
  and figure placement. Run the project's documented build if one exists.
- Record manuscript and research revisions, evidence used, assistance performed,
  checks completed, and remaining questions in the task record.
  Carnegie Mellon University Libraries' [LLM Documentation
  Guide](https://guides.library.cmu.edu/LLMDocumentationGuide) lists what such a
  record usually needs: the model and version, the settings, the prompts, and
  the iterations that led to the retained text.
- Check the current rules of your institution, collaborators, and target venue
  for permitted AI use and disclosure. Record any required disclosure from the
  work actually performed; this section does not establish a universal policy.
- Before supplying material, confirm that its confidentiality, consent, license,
  and applicable service terms permit that use. Local file access alone does not
  establish that processing stays on your machine.

The reviewable output is a manuscript change together with its evidence and
unresolved questions. Human authors remain responsible for accepting the text
and the scientific claims it makes.

## Further Reading

These are starting points for evaluation rather than endorsements. Read each one
against the evidence, privacy, and disclosure rules in this section; links were
checked in September 2026.

Reasoning effort and overthinking:

- Gaurav Srivastava et al., [Do LLMs Overthink Basic Math Reasoning? Benchmarking
  the Accuracy-Efficiency Tradeoff in Language Models](https://aclanthology.org/2026.findings-acl.1285/),
  Findings of ACL 2026
- Xinliang Frederick Zhang et al., [Do LLMs Really Need 10+ Thoughts for "Find the
  Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking](https://aclanthology.org/2026.acl-long.773/),
  ACL 2026 ([DeepMind summary](https://deepmind.google/research/publications/203490/))

LLM agents across the research workflow:

- Samuel Schmidgall et al., [Agent Laboratory: Using LLM Agents as Research
  Assistants](https://arxiv.org/pdf/2501.04227) — arXiv preprint. An end-to-end
  pipeline from literature review through a written report. Read it for which
  stages were automated and which human decisions were removed to make that
  possible.
- Chengwei Liu et al., [A Vision for Auto Research with LLM
  Agents](https://arxiv.org/html/2504.18765v1) — arXiv preprint. A proposed
  multi-agent framework spanning eight research phases: a position paper
  describing an intended system, not a validated one.
- [LLM Agents as AI Scientists: A Survey](https://openreview.net/forum?id=bfdUWy6rUA)
  — a readable overview of agent contributions to hypothesis discovery,
  experiment implementation, paper writing, and peer review. It is a student
  course project submitted to the
  [UIUC Spring 2025 CS598 LLM Agent Workshop](https://openreview.net/group?id=illinois.edu/UIUC/Spring_2025/CS598_LLM_Agent_Workshop),
  a class exercise that uses OpenReview to simulate peer review, so it is not
  peer-reviewed literature. Use it to find primary sources, and cite those
  instead.

Citation integrity and verification:

- William H. Walters and Esther Isabelle Wilder, [Fabrication and errors in the
  bibliographic citations generated by ChatGPT](https://doi.org/10.1038/s41598-023-41032-5),
  *Scientific Reports* 13:14045, 2023 — measures fabricated citations and
  substantive metadata errors across 636 generated citations.
- Mikaël Chelli et al., [Hallucination Rates and Reference Accuracy of ChatGPT
  and Bard for Systematic Reviews](https://doi.org/10.2196/53164), *Journal of
  Medical Internet Research*, 2024 — per-model hallucination rates on systematic
  review references, and why these tools should not be the primary means of
  assembling one.

Documenting and disclosing LLM use:

- Carnegie Mellon University Libraries, [LLM Documentation
  Guide](https://guides.library.cmu.edu/LLMDocumentationGuide) — practical
  guidance on recording prompts, models, and settings so assisted work stays
  reproducible and disclosable. It pairs with the task record in
  [Check Before Sharing](#check-before-sharing). Your own institution's and
  venue's rules still govern.

Multi-agent orchestration patterns:

- LlamaIndex, [Multi-Agent Report Generation using Agents as
  Tools](https://developers.llamaindex.ai/python/examples/agent/agents_as_tools/)
  — a worked orchestrator, research, writing, and review split. The role
  separation matches [Review with Separate Roles](#review-with-separate-roles),
  though report generation is a lower-stakes task than a manuscript.
- [`ndcorder/research-agent`](https://github.com/ndcorder/research-agent), listed
  on [Svelte Themes](https://sveltethemes.dev/ndcorder/research-agent) — a Claude
  Code toolkit that orchestrates agents to produce LaTeX papers. It is small and
  unvetted; inspect it before use, and do not let an autonomous drafting pipeline
  produce claims you have not checked against evidence.
- [`academic-pipeline`](https://claudemarketplaces.com/skills/imbad0202/academic-research-skills/academic-pipeline)
  by imbad0202 — a ten-stage orchestrator skill listed on a third-party
  marketplace that is not affiliated with Anthropic. Evaluate it with the checks
  in [External Skill Sets to Evaluate](#external-skill-sets-to-evaluate).

Vendor commentary:

- Suprmind, [Best AI for Writing Research Papers: A Multi-LLM Workflow That
  Holds](https://suprmind.ai/hub/insights/best-ai-for-writing-research-papers-a-multi-llm-workflow-that-holds/)
  — describes a multi-model drafting-and-critique workflow. It is marketing for
  the publisher's own product, so treat its model comparisons as claims rather
  than measurements.
