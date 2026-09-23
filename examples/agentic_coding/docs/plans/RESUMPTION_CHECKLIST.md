# Evidence-Based Resumption Checklist

Use this checklist at the start of a fresh session or before handing an active
task to another agent. It is a template for the target project's `docs/plans/`
directory; replace paths and commands with the project's own record.

- [ ] Read the active plan's status, objective, latest decision, evidence, and
  next action.
- [ ] Run `git status`, inspect the relevant `git diff`, and run
  `git log --oneline -n 10` from the repository root.
- [ ] Compare the current files, revision, and working tree with the plan's
  assumptions and acceptance criteria.
- [ ] Identify mismatches, missing artifacts, stale documentation, and checks
  that were not run.
- [ ] Report the current state, blockers, and next safe action before editing.
- [ ] Update the plan to correct any stale state before editing code.

Do not treat a previous chat transcript as evidence. If it contains a decision
or observed result needed by the next session, verify it against the repository
or recorded artifacts and then add the compact result to the active plan.
