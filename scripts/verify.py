"""Run checks and record their exact source identity; never reuse old logs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from signalfeed.provenance import code_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Requires a running isolated test service",
    )
    args = parser.parse_args()
    root = Path(os.environ["SIGNALFEED_HOME"])
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    identity = code_identity()
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        ["npm", "--prefix", "frontend", "run", "build"],
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
    ]
    env = dict(os.environ)
    if args.browser:
        import httpx

        health = httpx.get(
            env["SIGNALFEED_BASE_URL"] + "/api/health", trust_env=False
        ).json()
        if not health.get("test_events"):
            raise ValueError("Browser tests require an isolated test-event server")
        env["SIGNALFEED_BROWSER_REPORT"] = str(out / "browser-tests.json")
        commands.append(["npm", "--prefix", "frontend", "test"])
    results = []
    for i, command in enumerate(commands):
        log = out / f"validation-{i}.log"
        with log.open("w") as stream:
            completed = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        results.append(
            {"command": command, "exit_code": completed.returncode, "log": log.name}
        )
        print(results[-1], flush=True)
    report = {
        "run_id": root.name,
        "provenance": identity,
        "passed": all(r["exit_code"] == 0 for r in results)
        and identity == code_identity(),
        "commands": results,
    }
    if args.browser:
        report["browser"] = json.loads((out / "browser-tests.json").read_text())[
            "stats"
        ]
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
