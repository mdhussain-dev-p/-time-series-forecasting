from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
LOG_FOLDER = PROJECT_FOLDER / "logs"
LOG_FILE = LOG_FOLDER / "project_run.log"


def run_step(title: str, command: list[str], display_command: str) -> tuple[int, str]:
    result = subprocess.run(
        command,
        cwd=PROJECT_FOLDER,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout
    if result.stderr:
        output += "\n[Messages from the command]\n" + result.stderr
    print(f"\n{title}")
    print(display_command)
    print(output, end="" if output.endswith("\n") or not output else "\n")
    return result.returncode, output


def main() -> int:
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    package_names = ("numpy", "pandas", "matplotlib", "scikit-learn", "pytest")
    package_versions = []
    for package_name in package_names:
        try:
            package_versions.append(f"{package_name}=={importlib.metadata.version(package_name)}")
        except importlib.metadata.PackageNotFoundError:
            package_versions.append(f"{package_name}: not installed")

    sections = [
        "TIME-SERIES FORECASTING — PROJECT RUN LOG",
        f"Started: {started.isoformat(timespec='seconds')}",
        f"Python: {platform.python_version()}",
        "Packages: " + ", ".join(package_versions),
        "",
    ]

    steps = [
        ("Forecast generation", [sys.executable, "-m", "src.run_forecast"], "python -m src.run_forecast"),
        ("Automated tests", [sys.executable, "-m", "pytest", "-q"], "python -m pytest -q"),
    ]

    for title, command, display_command in steps:
        exit_code, output = run_step(title, command, display_command)
        sections.extend([
            title.upper(),
            "Command: " + display_command,
            output.rstrip(),
            f"Exit code: {exit_code}",
            "",
        ])
        if exit_code != 0:
            sections.append("Overall result: FAILED")
            LOG_FILE.write_text("\n".join(sections) + "\n", encoding="utf-8")
            print(f"\nThe run stopped. Details were saved to {LOG_FILE.relative_to(PROJECT_FOLDER)}.")
            return exit_code

    finished = datetime.now().astimezone()
    sections.extend([
        f"Finished: {finished.isoformat(timespec='seconds')}",
        "Overall result: SUCCESS",
        "Forecast files were refreshed in figures/ and results/.",
    ])
    LOG_FILE.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"\nBoth steps completed. Run log saved to {LOG_FILE.relative_to(PROJECT_FOLDER)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
