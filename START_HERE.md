# Start here — Time-Series Forecasting

This is the complete project folder. The dataset is included, so you do not
need to download a separate data file or set up MySQL.

## Put the files in your `Time Series` folder

1. Download the project ZIP.
2. Right-click the ZIP and choose **Extract All**.
3. Set your new `Time Series` folder as the extraction destination.
4. Open VS Code, choose **File → Open Folder**, and select `Time Series`.
5. Open **Terminal → New Terminal**. Make sure the terminal is PowerShell and
   that its current location is the project folder containing `src`, `data`,
   and `requirements.txt`.

## Set up Python on Windows

Run these commands in the VS Code PowerShell terminal, one at a time:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the following command in that terminal
and try the activation command again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the experiment and tests

```powershell
python -m src.run_forecast
python -m pytest -q
```

A successful test run should report **24 passed**. The forecasting command
prints the MAE and RMSE for both methods and refreshes the `figures` and
`results` folders.

## If something goes wrong

Do not guess at a fix or delete project files. Copy the complete error message
from the terminal and share it so the issue can be checked properly.

For an explanation of the experiment, open `README.md` and
`docs/PROJECT_WALKTHROUGH.md`.
