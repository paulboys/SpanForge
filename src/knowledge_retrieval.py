from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Any, List

# Simple retrieval over lexicon CSVs for context passed to LLM refinement.

def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get('term')]


def build_index(symptom_csv: Path, product_csv: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in _load_csv(symptom_csv) + _load_csv(product_csv):
        term = row['term'].strip().lower()
        canonical = (row.get('canonical') or term).strip()
        entry = index.setdefault(term, {"canonical": canonical, "examples": set()})
        entry["examples"].add(row.get('term', '').strip())
    # Convert sets to lists
    for v in index.values():
        v['examples'] = list(sorted(v['examples']))
    return index


def context_for_spans(spans: List[Dict[str, Any]], index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for s in spans:
        key = s.get('text', '').lower()
        if key in index:
            out[key] = index[key]
    return {"entries": out}
