"""Validated reporting for transformer efficiency experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def validate_efficiency_matrix(matrix: Mapping[str, Any]) -> None:
    """Reject incomplete matrices before they become portfolio evidence."""
    _require_mapping(matrix.get("configuration"), "configuration")
    _require_mapping(matrix.get("environment"), "environment")
    records = _require_list(matrix.get("records"), "records")
    comparisons = _require_list(matrix.get("comparisons"), "comparisons")
    if not isinstance(matrix.get("status"), str) or not matrix["status"].strip():
        raise ValueError("status must be a non-empty string")
    if not records:
        raise ValueError("records must not be empty")
    if not comparisons:
        raise ValueError("comparisons must not be empty")

    variants: set[tuple[str, str]] = set()
    for index, item in enumerate(records):
        record = _require_mapping(item, f"records[{index}]")
        try:
            key = (str(record["workload"]), str(record["variant"]))
            timing = _require_mapping(record["timing"], f"records[{index}].timing")
            median = float(timing["median_seconds"])
            throughput = float(record["throughput_per_second"])
            persistent = int(record["persistent_bytes"])
            workspace = int(record["workspace_bytes"])
            error = float(record["numerical_error"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"records[{index}] is incomplete") from exc
        if not all(key) or key in variants:
            raise ValueError("record workload/variant pairs must be unique and non-empty")
        if median <= 0 or throughput <= 0:
            raise ValueError("latency and throughput must be positive")
        if persistent < 0 or workspace < 0 or error < 0:
            raise ValueError("memory and numerical error must be non-negative")
        variants.add(key)

    for index, item in enumerate(comparisons):
        comparison = _require_mapping(item, f"comparisons[{index}]")
        try:
            workload = str(comparison["workload"])
            baseline = str(comparison["baseline"])
            candidate = str(comparison["candidate"])
            latency_ratio = float(comparison["latency_ratio"])
            throughput_ratio = float(comparison["throughput_ratio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"comparisons[{index}] is incomplete") from exc
        if (workload, baseline) not in variants or (workload, candidate) not in variants:
            raise ValueError("each comparison must reference measured variants")
        if latency_ratio <= 0 or throughput_ratio <= 0:
            raise ValueError("comparison ratios must be positive")


def create_report_snapshot(
    matrix: Mapping[str, Any], *, generated_at_utc: str | None = None
) -> dict[str, Any]:
    """Create a versioned report snapshot from one validated benchmark run."""
    validate_efficiency_matrix(matrix)
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Efficient Transformer Systems Report",
        "generated_at_utc": timestamp,
        "evidence": deepcopy(dict(matrix)),
        "claim_boundaries": [
            "Results describe one fixed-seed NumPy CPU microbenchmark run.",
            "Declared bytes cover managed persistent state and analytical workspace, not process peak memory.",
            "Throughput is comparable only within workloads sharing the same useful-work unit.",
            "Numerical error is not a language-model quality, accuracy, or perplexity measurement.",
            "CPU reference timings do not predict fused accelerator-kernel performance.",
        ],
    }


def _bytes(value: int) -> str:
    if value == 0:
        return "0 B"
    if value >= 1024**2:
        return f"{value / (1024**2):.3f}".rstrip("0").rstrip(".") + " MiB"
    if value >= 1024:
        return f"{value / 1024:.3g} KiB"
    return f"{value:,} B"


def render_systems_report(snapshot: Mapping[str, Any]) -> str:
    """Render Markdown directly from the machine-readable snapshot."""
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported report schema version")
    evidence = _require_mapping(snapshot.get("evidence"), "evidence")
    validate_efficiency_matrix(evidence)
    records = {
        (record["workload"], record["variant"]): record
        for record in evidence["records"]
    }
    lines = [
        "# Efficient Transformer Systems Report",
        "",
        f"Generated: `{snapshot['generated_at_utc']}`",
        "",
        "## Scope",
        "",
        str(evidence["status"])[0].upper() + str(evidence["status"])[1:] + ".",
        "",
        "## Comparable workload results",
        "",
        "| Workload | Candidate vs baseline | Median latency | Throughput | Persistent state | Workspace | Numerical error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in evidence["comparisons"]:
        workload = comparison["workload"]
        baseline = records[(workload, comparison["baseline"])]
        candidate = records[(workload, comparison["candidate"])]
        latency = (
            f"{candidate['timing']['median_seconds'] * 1000:.3f} ms vs "
            f"{baseline['timing']['median_seconds'] * 1000:.3f} ms "
            f"({comparison['latency_ratio']:.2f}x)"
        )
        throughput = (
            f"{candidate['throughput_per_second']:,.0f} vs "
            f"{baseline['throughput_per_second']:,.0f} "
            f"{candidate['work_unit']}/s"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(workload),
                    f"`{candidate['variant']}` vs `{baseline['variant']}`",
                    latency,
                    throughput,
                    f"{_bytes(candidate['persistent_bytes'])} vs {_bytes(baseline['persistent_bytes'])}",
                    f"{_bytes(candidate['workspace_bytes'])} vs {_bytes(baseline['workspace_bytes'])}",
                    f"{candidate['numerical_error']:.6g}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim boundaries", ""])
    for boundary in snapshot["claim_boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m pip install -e \".[dev]\"",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_systems_report.py",
            "python -m pytest -q",
            "```",
            "",
            "The Markdown report and JSON evidence are generated from the same in-memory snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_artifacts(
    snapshot: Mapping[str, Any], *, markdown_path: Path, json_path: Path
) -> None:
    """Write matching human-readable and machine-readable report artifacts."""
    markdown = render_systems_report(snapshot)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
