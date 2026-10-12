"""The forecasting worker's single entry point, run as a subprocess by the engine's job runner.

    python -m forecast_worker.run JOB.json OUT_DIR

JOB.json: {"kind": "profile"|"baseline", "inputs": {"<file name>": "<local path>"}, "params": {...}}
Writes its outputs into OUT_DIR and prints one JSON line: {"outputs": [...file names...], "summary": {...},
"questions": [...]} . It reads only the files it is given and writes only OUT_DIR; it never touches the engine's
database. Nothing here calls a model: every number is computed.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

WORKER_VERSION = "forecast-worker-1"


def _stage(inputs: dict[str, str]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="fw-in-"))
    for name, path in inputs.items():
        safe = Path(name).name                       # a file name, never a path from the job
        shutil.copyfile(path, tmp / safe)
    return tmp


COLUMNS_QUESTION = ("This file does not use the standard column names. Confirm what each column means: the date, the "
                    "columns that identify one series (or none: one series), the quantity to forecast, and when each row "
                    "becomes known. The profile SUGGESTS a mapping; nothing is read further until a person confirms it.")


def _mapped(job: dict, data: Path) -> dict:
    """Unfamiliar files become the standard copy the pipeline reads, from the person's confirmed mapping."""
    from .mapping import canonicalize
    return canonicalize(data, (job.get("params") or {}).get("answers") or {})


def run_profile(job: dict, out: Path) -> dict:
    from .mapping import needs_mapping, propose
    from .profile import profile
    data = _stage(job["inputs"])
    answers = (job.get("params") or {}).get("answers") or {}
    if needs_mapping(data) and not answers.get("columns"):
        prop = propose(data)
        (out / "profile.json").write_text(json.dumps({"stage": "mapping", **prop}, indent=1) + "\n")
        q = {"id": "Q-columns", "blocks": ["prepare", "train"], "question": COLUMNS_QUESTION, "evidence": prop}
        return {"outputs": ["profile.json"], "questions": [q],
                "summary": {"stage": "mapping", "files": [f["file"] for f in prop["files"]], "rows": sum(f["rows"] for f in prop["files"]),
                            "questions": ["Q-columns"], "series": None, "duplicate_keys": 0, "unparsed_rows": 0}}
    mapping = _mapped(job, data)
    result = profile(data)
    if mapping.get("mapped"):
        result["mapping"] = mapping
        # target, frequency and missing periods were confirmed with the column mapping: not asked twice
        answered = {"Q-mapping"} | ({"Q-missing-weeks"} if answers.get("missing_weeks") in ("missing", "zero") else set())
        result["questions"] = [q for q in result["questions"] if q["id"] not in answered]
    (out / "profile.json").write_text(json.dumps(result, indent=1) + "\n")
    return {"outputs": ["profile.json"], "questions": result["questions"],
            "summary": {"series": result["series_count"], "inputs_sha256": result["inputs_sha256"],
                        **({"mapped_rows_unread": sum(f["unread_dates"] + f["unread_quantities"] + f["unread_arrival_times"]
                                                      for f in result["mapping"]["files"])} if result.get("mapping") else {}),
                        "questions": [q["id"] for q in result["questions"]],
                        "duplicate_keys": len(result["duplicate_keys"]), "unparsed_rows": result["unparsed_rows"]}}


def run_baseline(job: dict, out: Path) -> dict:
    from .baseline import baseline
    data = _stage(job["inputs"])
    _mapped(job, data)
    return baseline(data, out, job.get("params") or {})


def run_evaluate(job: dict, out: Path) -> dict:
    from .evaluate import evaluate
    data = _stage(job["inputs"])
    _mapped(job, data)
    return evaluate(data, out, job.get("params") or {})


def run_issue(job: dict, out: Path) -> dict:
    from .operate import issue
    data = _stage(job["inputs"])
    _mapped(job, data)
    return issue(data, out, job.get("params") or {})


def run_plan_job(job: dict, out: Path) -> dict:
    from .operate import run_plan
    return run_plan(_stage(job["inputs"]), out, job.get("params") or {})


def run_context(job: dict, out: Path) -> dict:
    from .context.job import run
    return run(_stage(job["inputs"]), out, job.get("params") or {})


def run_scenario_job(job: dict, out: Path) -> dict:
    from .operate import run_scenario
    return run_scenario(_stage(job["inputs"]), out, job.get("params") or {})


def run_signals(job: dict, out: Path) -> dict:
    from .signals import run
    return run(_stage(job["inputs"]), out, job.get("params") or {})


def run_ingest(job: dict, out: Path) -> dict:
    from .ingest import run
    from .mapping import standardize_batches
    data = _stage(job["inputs"])
    answers = (job.get("params") or {}).get("answers") or {}
    if answers.get("columns"):                  # a sheet with its own column names: the confirmed mapping applies to every read
        standardize_batches(data, answers)
    return run(data, out, job.get("params") or {})


KINDS = {"signals": run_signals, "ingest": run_ingest, "scenario": run_scenario_job, "context": run_context, "profile": run_profile, "baseline": run_baseline, "evaluate": run_evaluate, "issue": run_issue, "plan": run_plan_job}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    kind = job.get("kind")
    if kind not in KINDS:
        print(json.dumps({"error": f"unknown job kind {kind!r}"}))
        return 2
    result = KINDS[kind](job, out)
    result.setdefault("summary", {})["worker_version"] = WORKER_VERSION
    import os
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # KB on Linux, bytes on macOS
    result["summary"]["peak_rss_mb"] = round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)
    if os.environ.get("VVDEX_CORRELATION_ID"):
        result["summary"]["correlation_id"] = os.environ["VVDEX_CORRELATION_ID"][:64]
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

