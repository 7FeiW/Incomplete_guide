# Git Workflow

## Never commit

**Do not run `git commit`.** Committing is the user's decision, always — including when the user
says "save the code", "save this", or asks for work to be written to disk. Those mean write the
files, not create a commit.

This applies even when:

- the change is small, or obviously correct
- tests pass and linting is clean
- a previous commit in the same session was requested explicitly
- the work is on a feature branch rather than `main`

Also do not run: `git push`, `git merge`, `git rebase`, `git reset --hard`, `git checkout` /
`git switch` onto another branch, or `git stash` — anything that rewrites history, moves the
branch pointer, or discards working-tree state.

### What to do instead

Leave the changes in the working tree and say what changed:

```
Wrote docs/foo_plan.md (new) and edited scripts/eval/bar.py — left uncommitted.
```

Do not stage with `git add` either, unless the user asks; an unexpectedly staged index is its own
surprise. If a commit seems warranted, offer the message and let the user run it.

### Allowed freely

Read-only inspection is fine and encouraged: `git status`, `git diff`, `git log`, `git show`,
`git blame`, `git ls-files`, `git check-ignore`, `git branch --list`.

### Rationale

Commits mark reviewed, intentional checkpoints in this repo's history. An agent-authored commit
inserts an unreviewed checkpoint into that record and, in an experiment-tracking codebase, can
attach a misleading provenance to results a run is later attributed to. Writing files is
reversible by inspection; committing is a claim about state that someone else has to undo.
