# Version Control and Git

> “I’m not a great programmer; I’m just a good programmer with great tools.” — Linus Torvalds

## Git clients and extensions:

- **Git (CLI)** — the canonical tool; learn the core commands below.
- **Git Fork** — GUI client: https://git-fork.com/
- **GitLens** — excellent VS Code extension: https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens

## Core Git Workflow For One Man Army

A simple, consistent trunk‑based workflow with lightweight feature branches is usually ideal for a solo project.

### Baseline: ultra‑simple solo loop

For small scripts, experiments, or throwaway tools, a single `main` branch is enough. Core loop:

1. `git status` to see changes.
2. `git add -p` or `git add .` to stage.
3. `git commit -m "short, action‑based message"`.
4. `git push origin main` as your offsite backup and sync point.

This works well until you have “real users”, releases, or long‑running features.

### Recommended core workflow for serious solo projects

Use trunk‑based development with short‑lived feature/bugfix branches and keep `main` always deployable.

- Long‑lived branches:
	- `main`: always in a releasable state; tags mark releases.
	- Optional `dev`: only if you often stage multiple features before a release.
- Short‑lived branches:

```
feat/<slug>  # features
fix/<slug>   # bugfixes
```

Typical cycle:

1. `git checkout -b feat/new-parser main`
2. Work and commit in small, logical chunks.
3. Optionally `git rebase -i main` to clean history before merge.
4. `git checkout main && git merge --no-ff feat/new-parser` (or fast‑forward).
5. `git tag v0.3.0` for a release and `git push --all --tags`.
6. `git branch -d feat/new-parser` to keep things tidy.

### When to add a `dev` branch and a release branch

Add a `dev` branch if you:

- Have a production deployment and want `main` to track only what is live, with `dev` for the “next release”.
- Need to batch multiple features into a release or maintain hotfixes against older tags.

Pattern with `dev`:

- `main` → production (tagged releases)
- `dev` → integrates features for the next release
- `feat/*` / `fix/*` → branch from `dev`, merge into `dev`, then periodically merge `dev` → `main` and tag

### Practical solo habits that matter most

- Keep `main` buildable and deployable; branch for risky or long‑running work.
- Commit often with messages that describe the change, not the file list.
- Tag releases; if deployment breaks, check out the tag to patch.
- Regularly prune merged branches: `git branch --merged` then delete merged feature branches.

If you tell me how you ship (library vs web app vs scripts, CI or not), I can add a tailored template with exact commands.

## Core Git Workflow For Teamwork

For teams, follow the branching and PR workflow already documented above: create a branch per task, rebase/merge responsibly, and use PRs for code review. The following commands are useful when collaborating:

1. Create a branch per task/feature: `git checkout -b feat/short-description`
2. Make small, focused commits with clear messages:

```
git add <files>
git commit -m "Short: one-line summary\n\nLonger description explaining why"
```

3. Keep your branch up to date with the remote main branch:

```
git fetch origin
git rebase origin/main   # or `git merge origin/main` depending on your team's policy
```

4. Push your branch and open a pull/merge request:

```
git push -u origin feat/short-description
```

## Branching:

- Use short, descriptive branch names: `feat/`, `fix/`, `chore/`, `docs/`.
- Prefer rebasing local feature branches onto main for a linear history, but avoid rebasing shared branches.
- Squash commits for small fixup commits when merging if your team prefers a cleaner main branch.

Common commands quick reference:

- See recent commits: `git log --oneline --graph --decorate --all`
- See local changes: `git status` and `git diff`
- Undo a file: `git checkout -- path/to/file` (or `git restore path/to/file` with newer Git)
- Create tag for release: `git tag -a v1.2.0 -m "Release v1.2.0"` then `git push origin v1.2.0`

Working with remotes:

- Add remote: `git remote add upstream https://github.com/owner/repo.git`
- Fetch from remote: `git fetch upstream`
- Cherry-pick a commit: `git cherry-pick <commit-hash>`

Collaborating safely:

- Use signed commits (`git commit -S`) if your project requires verification.
- Add a good `.gitignore` to avoid committing build artifacts, secrets, and large files.
- Use protected branches on Git hosting (require PR reviews, CI) for `main`/`master`.

