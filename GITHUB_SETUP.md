# Publishing NumPyNet carefully

Confirm the repository destination, working tree, and commit identity before publishing.

## If this is your first local Git repository

Create an **empty** `NumPyNet` repository in the `tkarimdk` GitHub account (no generated README, `.gitignore`, or license). Decide whether it should be public or private; GitHub visibility is your choice.

Before the first push, **choose the code license policy**: no `LICENSE` means other users generally do not receive reuse permissions; an explicit MIT/Apache/GPL license grants different rights. Do not add a license without understanding the implications.

Confirm that Git is installed and configured with **your own GitHub-associated identity** (`git config user.name` and `git config user.email`). In PowerShell, from the folder containing `pyproject.toml`:

```powershell
git init -b main
git add .
git status
git diff --cached --stat
git commit -m "feat: add NumPyNet stability diagnostics and recovery experiments"
git remote add origin https://github.com/tkarimdk/NumPyNet.git
git push -u origin main
```

This publishes the v0.4 snapshot as the initial commit.

## If a previous version is already committed locally

Do not initialize another Git repository. **Back up your project first**. Copy the updated source, tests and docs from the v0.4 ZIP into your existing tracked repository; do not copy the `.venv` environment. In the existing repository:

```powershell
git status
git diff
git add pyproject.toml src tests docs examples README.md CHANGELOG.md ROADMAP.md GITHUB_SETUP.md .gitignore
git commit -m "feat: compare gradient clipping schedules and backtracking"
git push
```

Review `git diff` and `git status` before every commit and confirm you're on the right repository and branch. Never commit `.venv`, private files, or unreviewed `results/` output.

## Meaningful release checklist

- All tests pass (`python -m unittest discover -s tests -v`).
- You understand at least one major new feature and can explain the key design decisions.
- Experiments are reproducible with documented input, seed, update budget and limitations.
- `docs/learning-log.md` includes the observed result and your own interpretation.
- Commit or merge only when the code is functional; activity follows useful engineering work.


## Quick verification before pushing

From the project root (the folder containing `pyproject.toml`), run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m numpynet check-gradients
.\.venv\Scripts\python.exe -m numpynet stabilize-xor --modes baseline clip decay backtrack --seeds 7 8 9 --epochs 2000
```

`results/` will be created locally and ignored by Git. Check `git status` carefully before pushing: the source, tests, docs, workflow, and curated `examples/` are intended for release; virtual environments, credentials, and generated output are not.

After pushing, check the **Actions** tab. A passing local test run does **not** guarantee GitHub Actions will succeed until its workflow has run on GitHub.
