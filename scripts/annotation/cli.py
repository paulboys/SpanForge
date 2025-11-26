#!/usr/bin/env python
"""Unified CLI wrapper for annotation workflow.

Subcommands:
  bootstrap       -> create Label Studio project payload
  import-weak     -> convert weak JSONL to tasks (+optional push)
  quality         -> compute annotation quality metrics
  adjudicate      -> consensus across multiple gold files
  register        -> append batch provenance to registry
  refine-llm      -> refine weak labels using LLM (OpenAI, Azure, Anthropic)
  evaluate-llm    -> evaluate weak → LLM → gold with comprehensive metrics
  plot-metrics    -> generate visualizations from evaluation reports

Example:
  python scripts/annotation/cli.py bootstrap --name "Adverse Event NER"
  python scripts/annotation/cli.py evaluate-llm --weak weak.jsonl --refined llm.jsonl --gold gold.jsonl --output report.json
  python scripts/annotation/cli.py plot-metrics --report report.json --output-dir plots/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SUBCMDS = {
    "bootstrap": "init_label_studio_project.py",
    "import-weak": "import_weak_to_labelstudio.py",
    "quality": "quality_report.py",
    "adjudicate": "adjudicate.py",
    "register": "register_batch.py",
    "refine-llm": "refine_llm.py",
    "evaluate-llm": "evaluate_llm_refinement.py",
    "plot-metrics": "plot_llm_metrics.py",
}


def run_script(script: str, args: list[str]):
    script_path = SCRIPT_DIR / script
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Annotation workflow CLI")
    parser.add_argument("subcommand", choices=SUBCMDS.keys(), help="Action to perform")
    args, remainder = parser.parse_known_args()
    script = SUBCMDS[args.subcommand]
    code = run_script(script, remainder)
    if code != 0:
        print(f"Subcommand '{args.subcommand}' exited with code {code}", file=sys.stderr)
        sys.exit(code)


if __name__ == "__main__":
    main()
