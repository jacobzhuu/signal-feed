"""Verify the editable lock entry resolves to the checkout being installed."""

import json
from pathlib import Path
import shutil
import subprocess
import sys


def test_editable_install_from_another_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "another checkout"
    checkout.mkdir()
    for name in ["pyproject.toml", "requirements.lock.txt"]:
        shutil.copy2(root / name, checkout / name)
    shutil.copytree(
        root / "src",
        checkout / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    editable = [
        line
        for line in (checkout / "requirements.lock.txt").read_text().splitlines()
        if line.startswith("-e ")
    ]
    assert editable == ["-e ."]
    # Reuse installed third-party dependencies; actually build/install only the
    # local project in an isolated environment, without downloading GPU wheels.
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True
    )
    python = venv / "bin/python"
    local_lock = checkout / "local-requirements.txt"
    local_lock.write_text("\n".join(editable) + "\n")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-r",
            str(local_lock),
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json,signalfeed; print(json.dumps(signalfeed.__file__))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(json.loads(result.stdout)).is_relative_to(checkout)
