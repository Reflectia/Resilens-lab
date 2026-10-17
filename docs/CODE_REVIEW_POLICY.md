# Code Review Policy

Short rules so every change is reviewed the same way. Applies to everyone, including the team lead.

## 1. Workflow

1. Never commit to `main`. Work in a branch named `DEV_T<id>_<short_name>` (e.g. `DEV_T19_Basic_setup`).
2. Keep PRs small: one task, ideally under ~400 changed lines. Split larger work.
3. Open a PR into the integration branch `INT_RESILENSE_2.0` (not `main`; `main` is updated only from `INT_RESILENSE_2.0` by the team lead) using the PR template. Use **Draft** until it is ready for review.
4. Author self-reviews the diff first and ticks the author checklist.
5. A PR needs at least **1 approval** before merge.
6. A PR is merged only when CI (GitHub Actions) is green: `CI` (`ruff check .`, `ruff format --check .`, `pytest tests`) and `PR title` (format `T-<id>: ...`).
7. The author merges their own PR only after approval, using **Squash and merge**; delete the branch.

## 2. Review timing (SLA)

- First response within **1 working day**.
- Author answers comments within 1 working day. Reviewer re-checks within 1 working day.

## 3. PR titles and commits

The **PR title** is what matters: with squash merge it becomes the commit message on the integration branch.

Format: `T-<id>: <imperative summary, max ~72 chars>`

Good: `T-21: add confidence threshold validation to YOLOX config`
Bad: `fix`, `changes`, `wip`, `final final 2`

Rules:
- Commits inside a PR are free-form but meaningful (no `wip` spam); they are squashed anyway.
- No secrets, model weights, videos, or generated files (`runs/`, `data/`, `weights/`).
- Repo setting (Settings → General → Pull Requests): squash merge default message = **Pull request title**, otherwise single-commit PRs use the commit message.
- Release PR `INT_RESILENSE_2.0` → `main` may be titled `Release ...`.

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
- Don't force-push after review has started without saying so; prefer new commits (squash handles history).
- Keep the PR description accurate as it changes.

## 7. Repo settings and enforcement

The free GitHub plan does not enforce branch protection on private repos, so the rules above are a team agreement that GitHub does not check for us: nobody merges a PR with a red CI or without an approval, and nobody pushes directly to `INT_RESILENSE_2.0` or `main`. Free settings that still help (Settings → General → Pull Requests):

- Allow only **Squash merging**; default commit message = **Pull request title**.
- Enable **Automatically delete head branches**.
