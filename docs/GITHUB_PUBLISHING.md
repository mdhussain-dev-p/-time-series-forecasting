# Publish updates to GitHub

The repository is already published. Use the following process when publishing later changes.

## Update the local copy first

```powershell
git status --short
git pull --ff-only origin main
```

If `git status --short` lists changes you have not committed, review them before pulling.

## Run the project checks

```powershell
python -m pip install -r requirements.lock
python -m src.run_forecast
python -m pytest
```

## Review and commit the changes

```powershell
git status --short
git add .
git diff --cached --check
git commit -m "Improve forecasting baselines and evaluation"
git push origin main
```

Do not add `.venv/`, local editor settings, credentials or private data. The `.gitignore` file excludes the virtual environment and common local files.
