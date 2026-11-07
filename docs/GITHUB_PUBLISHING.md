# Publish the project on GitHub

This folder is ready to become a Git repository. The ZIP intentionally does
not contain a `.git` directory; Git will create that on your computer.

## 1. Create an empty repository on GitHub

Create a public repository named `time-series-forecasting`. Leave the options
to add a README, `.gitignore` or licence unchecked because those files are
already included here.

## 2. Open the project terminal

In VS Code, open the `Time Series` folder. Select **Terminal → New Terminal**
and make sure the terminal is in the folder that contains `README.md`, `src`
and `data`.

## 3. Create the local Git repository

Run these commands one by one in PowerShell:

```powershell
git init -b main
git add .
git status
git commit -m "Add monthly temperature forecasting project"
```

Before committing, check the `git status` output. The project files should be
listed. The `.venv` folder, if created, should be ignored automatically.

## 4. Connect and push

Replace the example URL with the URL of your own empty GitHub repository:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/time-series-forecasting.git
git push -u origin main
```

If GitHub asks you to authenticate, complete its normal browser sign-in flow.
Do not paste account passwords or access tokens into source files or README
text.

## 5. Check the published repository

Refresh the repository page and check that it contains `README.md`, `src`,
`tests`, `data`, `figures`, `results` and `docs`. Open the README on GitHub to
make sure the project overview and results display correctly.
