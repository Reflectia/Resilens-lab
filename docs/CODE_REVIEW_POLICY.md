# Code Review Policy

Short rules so every change is reviewed the same way. Applies to everyone, including the team lead.

## 1. Workflow

1. Never commit to `main`. Work in a branch named `DEV_T<id>_<short_name>` (e.g. `DEV_T19_Basic_setup`).
2. Keep PRs small: one task, ideally under ~400 changed lines. Split larger work.
3. Open a PR into `main` using the PR template. Use **Draft** until it is ready for review.
4. Author self-reviews the diff first and ticks the author checklist.
5. A PR needs at least **1 approval** before merge.
6. A PR is merged only when CI (GitHub Actions) is green: `CI` (`ruff check .`, `ruff format --check .`, `pytest tests`) and `PR title` (format `T-<id>: ...`).
7. The author merges their own PR only after approval, using **Create a merge commit**; delete the branch.

## 2. Review timing (SLA)

- First response within **1 working day**.
- Author answers comments within 1 working day. Reviewer re-checks within 1 working day.

## 3. PR titles and commits

The **PR title** is what matters: it goes into the merge commit message on `main`.

Format: `T-<id>: <imperative summary, max ~72 chars>`. Subtasks use the parent task id: task T-04-0 is titled `T-04: ...`.

Good: `T-21: add confidence threshold validation to YOLOX config`
Bad: `fix`, `changes`, `wip`, `final final 2`

Rules:
- Commits inside a PR stay in the history of `main`, so each one needs a meaningful message (no `wip` spam).
- No secrets, model weights, videos, or generated files (`runs/`, `data/`, `weights/`).

## 4. Reviewer checklist

**Correctness**
- [ ] Does it solve the task/issue it links to? Does it do anything beyond it?
- [ ] Edge cases and error handling (empty input, bad video, missing file/config)?
- [ ] No obvious bugs, race conditions, or resource leaks (files, video captures, GPU memory).

**Design & readability**
- [ ] Fits the existing structure (`core`, `detection`, `enhancement`, `config`, …); no duplicated logic.
- [ ] Names are clear; functions are small; comments explain *why*, not *what*.
- [ ] No dead code, debug prints, or commented-out blocks.

**Tests**
- [ ] New behavior or bug fix has tests; tests are meaningful, not just for coverage.
- [ ] `ruff check .`, `ruff format --check .` and `pytest` pass locally and in CI.

**Config & reproducibility** (important for this project)
- [ ] Experiment settings live in `configs/`, not hard-coded.
- [ ] Results stay reproducible (seeds, versions, config saved with the run).
- [ ] Config/schema changes are backward compatible or documented.

**Security & safety**
- [ ] No secrets, tokens, or personal paths committed.
- [ ] API inputs/uploads validated (file type, size, path traversal).
- [ ] New dependencies are justified and added to `pyproject.toml`.

**Docs**
- [ ] README/docs updated if commands, setup, or behavior changed.

## 5. How to give feedback

- Review the code, not the person. Ask questions ("what happens if…?") rather than commanding.
- Prefix comments: `blocker:` must fix before merge · `suggestion:` optional · `nit:` tiny style point · `question:` need understanding.
- Approve when only `nit`/`suggestion` items remain. Request changes only for blockers.
- Praise good solutions too.

## 6. Author responsibilities

- Respond to every comment (fix it or explain why not); reviewer resolves the thread.
- Don't force-push after review has started without saying so; prefer new commits.
- Keep the PR description accurate as it changes.

## 7. Repo settings and enforcement

The repository is public, so GitHub branch protection is available on the free plan. Protect `main` (Settings → Branches → Add branch protection rule):

- Require a pull request before merging, with **1 approval**.
- Require the status checks `CI` and `PR title` to pass.
- Do not allow force pushes or deletions.

Settings → General → Pull Requests:

- Allow **merge commits**; default message = **Pull request title**.
- Enable **Automatically delete head branches**.
