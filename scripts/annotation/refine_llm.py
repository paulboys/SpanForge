#!/usr/bin/env python
"""Refine weak labels with (stub) LLM suggestions.

Usage:
  python scripts/annotation/refine_llm.py --weak data/output/weak.jsonl --out data/output/refined_weak.jsonl \
      --prompt prompts/boundary_refine.txt --mode boundary --dry-run

Currently uses stub LLM (no external calls)."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import Dict, Any, List

import sys
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # type: ignore
from src.llm_agent import LLMAgent  # type: ignore
from src.knowledge_retrieval import build_index, context_for_spans  # type: ignore

CONF_MID_LOW = 0.55
CONF_MID_HIGH = 0.75

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def load_weak(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def filter_uncertain_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in spans:
        conf = float(s.get('confidence', 1.0))
        if CONF_MID_LOW <= conf <= CONF_MID_HIGH:
            out.append(s)
    return out


def refine(records: List[Dict[str, Any]], prompt_template: str, index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    agent = LLMAgent()
    refined = []
    for r in records:
        text = r.get('text', '')
        orig_spans = r.get('spans', [])
        target_spans = filter_uncertain_spans(orig_spans)
        knowledge = context_for_spans(target_spans, index)
        suggestions = agent.suggest(prompt_template, text, target_spans, knowledge)
        r['llm_suggestions'] = [
            {
                'start': s.start,
                'end': s.end,
                'label': s.label,
                'negated': s.negated,
                'canonical': s.canonical,
                'confidence_reason': s.confidence_reason,
                'llm_confidence': s.llm_confidence,
                'source': 'llm_refine'
            } for s in suggestions
        ]
        r['llm_meta'] = {
            'prompt_version': get_config().llm_prompt_version,
            'model': get_config().llm_model,
            'provider': get_config().llm_provider,
            'text_hash': _hash_text(text),
            'suggestion_count': len(r['llm_suggestions'])
        }
        refined.append(r)
    return refined


def main():
    ap = argparse.ArgumentParser(description='LLM refinement (stub)')
    ap.add_argument('--weak', required=True, help='Input weak JSONL path')
    ap.add_argument('--out', required=True, help='Output refined JSONL path')
    ap.add_argument('--prompt', required=True, help='Prompt template file')
    ap.add_argument('--dry-run', action='store_true', help='Do not write output file')
    args = ap.parse_args()
    weak_path = Path(args.weak)
    out_path = Path(args.out)
    prompt_path = Path(args.prompt)
    template = prompt_path.read_text(encoding='utf-8')

    # Build index (symptom/product lexicons)
    index = build_index(Path('data/lexicon/symptoms.csv'), Path('data/lexicon/products.csv'))
    records = load_weak(weak_path)
    refined_records = refine(records, template, index)
    if not args.dry_run:
        save_jsonl(refined_records, out_path)
    print(f"Refined {len(refined_records)} records; wrote suggestions to {out_path}" if not args.dry_run else "Dry run complete.")

if __name__ == '__main__':
    main()
