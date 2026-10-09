import json

import pytest

from transformer_systems import (
    create_report_snapshot,
    render_systems_report,
    write_report_artifacts,
)


def matrix() -> dict[str, object]:
    def record(variant: str, median: float, persistent: int, workspace: int):
        return {
            "workload": "decode_s128",
            "variant": variant,
            "useful_work": 1,
            "work_unit": "generated_tokens",
            "throughput_per_second": 1 / median,
            "persistent_bytes": persistent,
            "workspace_bytes": workspace,
            "managed_bytes": persistent + workspace,
            "numerical_error": 0.0,
            "timing": {"median_seconds": median},
        }

    return {
        "status": "measured fixture",
        "configuration": {"seed": 21},
        "environment": {"python": "test"},
        "records": [
            record("mha", 0.002, 400, 20),
            record("gqa", 0.001, 100, 20),
        ],
        "comparisons": [
            {
                "workload": "decode_s128",
                "baseline": "mha",
                "candidate": "gqa",
                "latency_ratio": 0.5,
                "throughput_ratio": 2.0,
                "persistent_bytes_ratio": 0.25,
                "workspace_bytes_ratio": 1.0,
            }
        ],
    }


def test_report_renders_values_from_the_snapshot() -> None:
    snapshot = create_report_snapshot(
        matrix(), generated_at_utc="2026-10-09T18:00:00+00:00"
    )

    markdown = render_systems_report(snapshot)

    assert "1.000 ms vs 2.000 ms (0.50x)" in markdown
    assert "1,000 vs 500 generated_tokens/s" in markdown
    assert "100 B vs 400 B" in markdown
    assert "2026-10-09T18:00:00+00:00" in markdown


def test_written_markdown_and_json_share_one_snapshot(tmp_path) -> None:
    snapshot = create_report_snapshot(matrix(), generated_at_utc="fixed")
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    write_report_artifacts(
        snapshot, markdown_path=markdown_path, json_path=json_path
    )

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved == snapshot
    assert "`gqa` vs `mha`" in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.pop("records"), "records"),
        (lambda value: value.update(records=[]), "records"),
        (
            lambda value: value["comparisons"][0].update(candidate="missing"),
            "measured variants",
        ),
        (
            lambda value: value["records"][1]["timing"].update(
                median_seconds=0
            ),
            "latency",
        ),
    ],
)
def test_snapshot_rejects_incomplete_evidence(mutation, message) -> None:
    value = matrix()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        create_report_snapshot(value)
