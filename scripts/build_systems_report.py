"""Generate matching Markdown and JSON evidence from one benchmark matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from transformer_systems import create_report_snapshot, write_report_artifacts


ROOT = Path(__file__).resolve().parents[1]


def capture_matrix() -> dict[str, object]:
    environment = os.environ.copy()
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_efficiency_matrix.py")],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    snapshot = create_report_snapshot(capture_matrix())
    write_report_artifacts(
        snapshot,
        markdown_path=ROOT / "artifacts" / "SYSTEMS_REPORT.md",
        json_path=ROOT / "artifacts" / "efficiency_snapshot.json",
    )
    print(json.dumps({
        "markdown": "artifacts/SYSTEMS_REPORT.md",
        "json": "artifacts/efficiency_snapshot.json",
        "workloads": len(snapshot["evidence"]["comparisons"]),
    }, indent=2))


if __name__ == "__main__":
    main()
