#!/usr/bin/env python
"""Convert weak labels JSONL into Label Studio task import JSON.

Weak JSONL format (one line per record):
{"text": ..., "spans": [{"start":...,"end":...,"label":...}]}

Output: tasks.json containing array of {"data":{"text":...}} optionally with pre-annotations.
By default we DO NOT include pre-annotations to reduce bias; enable via --include-preannotated.

Usage:
  python scripts/annotation/import_weak_to_labelstudio.py --weak data/output/notebook_test.jsonl --out data/annotation/exports/tasks.json

Optional API push (if --push and LABEL_STUDIO_API_KEY + --project-id provided).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import sys
import requests  # type: ignore
from dotenv import load_dotenv
load_dotenv()  # Load .env file from project root

def read_weak(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def to_tasks(records, include_preannotated: bool, min_confidence: float | None):
    tasks = []
    for r in records:
        text = r.get("text") or r.get("data", {}).get("text")
        if not text:
            continue
        spans = r.get("spans") or r.get("weak_spans") or []
        if min_confidence is not None:
            spans = [s for s in spans if s.get("confidence", 1.0) >= min_confidence]
        task = {"data": {"text": text}}
        if include_preannotated and spans:
            # Label Studio expects 'annotations' or import via predictions. Here we prepare 'predictions'.
            task["predictions"] = [
                {
                    "model_version": "weak_v1",
                    "result": [
                        {
                            "value": {
                                "start": s["start"],
                                "end": s["end"],
                                "text": s["text"],
                                "labels": [s["label"]],
                            },
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for s in spans
                    ],
                }
            ]
        tasks.append(task)
    return tasks


def push_tasks(tasks, project_id: int, base_url: str, api_key: str):
    url = f"{base_url.rstrip('/')}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {api_key}"}
    # Label Studio expects tasks array directly, not wrapped in {"tasks": ...}
    resp = requests.post(url, headers=headers, json=tasks)
    if resp.status_code >= 300:
        raise SystemExit(f"Import failed: {resp.status_code} {resp.text}")
    print(f"Imported {len(tasks)} tasks to project {project_id}")


def main():
    parser = argparse.ArgumentParser(description="Prepare (and optionally import) weak tasks into Label Studio")
    parser.add_argument("--weak", required=True, help="Weak labels JSONL path")
    parser.add_argument("--out", required=True, help="Output tasks JSON path")
    parser.add_argument("--include-preannotated", action="store_true", help="Include weak spans as predictions")
    parser.add_argument("--min-confidence", type=float, default=None, help="Filter spans below confidence threshold")
    parser.add_argument("--push", action="store_true", help="Push tasks to Label Studio via API")
    parser.add_argument("--project-id", type=int, default=None, help="Target project id (required if --push)")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Label Studio base URL")
    args = parser.parse_args()

    weak_path = Path(args.weak)
    out_path = Path(args.out)
    records = read_weak(weak_path)
    tasks = to_tasks(records, include_preannotated=args.include_preannotated, min_confidence=args.min_confidence)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {out_path}")

    if args.push:
        if args.project_id is None:
            raise SystemExit("--project-id required when using --push")
        api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        if not api_key:
            raise SystemExit("LABEL_STUDIO_API_KEY env var required for push")
        try:
            push_tasks(tasks, args.project_id, args.base_url, api_key)
        except Exception as e:  # pragma: no cover
            print(f"Import error: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
