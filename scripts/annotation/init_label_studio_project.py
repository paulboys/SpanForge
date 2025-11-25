#!/usr/bin/env python
"""Bootstrap a local Label Studio project for NER.

Steps performed:
1. Ensures LABEL_STUDIO_DISABLE_TELEMETRY=1 for privacy.
2. Verifies label config exists (creates minimal if missing).
3. Prints curl instructions to create a project via API (requires LS running and API key).

Usage:
  python scripts/annotation/init_label_studio_project.py --name "Adverse Event NER" --label-config data/annotation/config/label_config.xml

This script does NOT start Label Studio. Start it separately:
  label-studio start --no-browser

API Docs: https://labelstud.io/guide/api
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_CONFIG = """<View>\n  <Labels name=\"label\" toName=\"text\">\n    <Label value=\"SYMPTOM\" background=\"red\"/>\n    <Label value=\"PRODUCT\" background=\"blue\"/>\n  </Labels>\n  <Text name=\"text\" value=\"$text\"/>\n</View>\n"""


def ensure_config(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Label Studio project configuration")
    parser.add_argument("--name", required=True, help="Project name")
    parser.add_argument(
        "--label-config",
        default="data/annotation/config/label_config.xml",
        help="Path to label config XML",
    )
    parser.add_argument(
        "--output",
        default="data/annotation/config/project_bootstrap.json",
        help="Where to write project creation payload",
    )
    args = parser.parse_args()

    # Disable telemetry for current process and remind user for shell
    os.environ["LABEL_STUDIO_DISABLE_TELEMETRY"] = "1"

    config_path = ensure_config(Path(args.label_config))

    payload = {
        "title": args.name,
        "label_config": config_path.read_text(encoding="utf-8"),
        "description": "Adverse event NER annotation project (SYMPTOM, PRODUCT).",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Project payload written to {out_path}")
    print("\nNext: obtain API key from Label Studio UI (Account Settings) and create project:")
    print("PowerShell example (replace <API_KEY> and <PORT>):")
    print(
        r"curl -X POST http://localhost:8080/api/projects -H \"Authorization: Token <API_KEY>\" -H \"Content-Type: application/json\" --data @data/annotation/config/project_bootstrap.json"
    )
    print("Then import tasks via import_weak_to_labelstudio.py or UI upload.")


if __name__ == "__main__":
    main()
