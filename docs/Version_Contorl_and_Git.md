# Version Control and Git

> “I’m not a great programmer; I’m just a good programmer with great tools.” — Linus Torvalds

## Git clients and extensions:

- **Git (CLI)** — the canonical tool; learn the core commands below.
- **Git Fork** — GUI client: https://git-fork.com/
- **GitLens** — excellent VS Code extension: https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens

## Git Workflow For Research Project
Research Project share a lot of commonity of other software develpment project, however:
- Researchers do not deploy multiple times a day. We generate analyses, not production artifacts.
- Researchers do not have a fixed release plan, we are must like only create a a few release once project is finished or reach a milestone.
- Experiments produce figures, models, papers, they are not "merge immediately then deploy"
- Research tasks often need:
    - Multiple iteration cycles
    - Quick integration
    - Minimal isolation
    - Tags for preprint, revision, dataset versions

### Use Trunk-Based Git Workflow for Research Project
Uses a **trunk-based development workflow**.  That is All work is done in **short-lived branches** that merge frequently into the mainline branch (**`main`**) - a.k.a trunk. 

Core Idea:
- **Single trunk (`main`)**: Always deployable and reproducible; no long-lived branches.
- **Short-lived branches**: <1 day lifespan, 1 developer per branch.

This workflow is gear towards to ensure
- Continuous reproducibility  
- Minimal merge conflicts
- Incremental scientific development  
- Clean and traceable version history 

### Branching Model

**main** Branch：
- The single authoritative branch.  
- Must remain **runnable and stable** at all times.  
- All contributions merge into `main` through reviewed pull requests.

**develop** Branch
- In the case that there is a deployment requirement, keep main as the **runnable and stable**
- Use this branch to integrate latest change from each feature branch

**Short-Lived Branches** branch：
- Every task should occur in its own temporary branch.
- These branch should be small in scope and focused on one change
- These branch shoul be Merged within a few days and Deleted after merging

**Long-Lived Branches** branch:
- Idealy we should not have any long lived branch
- Use this to store refactoring code
- DO NOT use this to store new features or new task, as switch between branch would be hard to deal with in a day-to-day base.

### Commit 

Don’t Commit Large Data Files, Git slows down if large binary files are added.
Use:
- cloud storage
- data servers

Commit only:
- metadata
- configs
- scripts

Commit Often, Use concise, descriptive commit messages:

Commit Message examples:
- `Add dropout hyperparameter to model config`
- `Fix off-by-one error in data indexing`
- `Implement LR sweep experiment`
- `Update methods text in manuscript`

Avoid vague messages like “fix stuff” or “update file”.

### Use Tags for “Published” States

Use Git tags to mark important scientific or development milestones:
```
git tag -a v0.1-preprint -m "Version for preprint submission"
git tag -a v1.0-paper -m "Final version matching accepted manuscript"
git push origin --tags
```

### Pull Request

PR requirements:
- Code runs without breaking existing functionality  
- Clear commit messages  
- Reviewer approval (if working collaboratively) 

### Collaboration Guidelines

- Never push directly to main
- Always use a branch + PR
- Keep PRs small and focused
- Document experimental branches
- Tag important analysis states
- Use Issues to track tasks and bugs


## Why Not Use a Feature-Based Workflow?

A traditional feature-based workflow (e.g., long-lived feature branches, Git Flow) is common in enterprise software but poorly suited for scientific research. In this model, developers create large, isolated branches that may live for weeks or months before merging. Although this approach can work for highly structured product development, it introduces several problems in computational research environments.


## Why Not Use a GitHub WorkFlow?

GitHub Flow is a lightweight branching model commonly used for web development and continuous deployment. It emphasizes simplicity and rapid integration. The core idea is that all work happens on short-lived branches that branch off from main. Once changes are tested and reviewed through a pull request, they are merged back into main and deployed immediately. Although GitHub Flow resembles trunk-based development, there are key differences in expectations and usage patterns.