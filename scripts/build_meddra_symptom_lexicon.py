#!/usr/bin/env python
"""Build MedDRA-based symptom lexicon from PT list and synonym mappings.

USAGE (PowerShell):
    python scripts/build_meddra_symptom_lexicon.py \
        --pt-file data/meddra/pt.csv \
        --syn-file data/synonyms/symptom_synonyms.csv \
        --out-file data/lexicon/symptoms.csv

Inputs:
  pt.csv:        pt_name,pt_code               (Licensed MedDRA subset you supply)
  symptom_synonyms.csv: synonym,pt_name        (Consumer or CHV variants mapping to PT)

Output columns:
  term,canonical,source,concept_id
    - term: surface form encountered in text
    - canonical: MedDRA Preferred Term (PT)
    - source: provenance tag (e.g., 'meddra_pt','user_syn','chv')
    - concept_id: PT code (if provided) else blank

NOTE: Script will not ship MedDRA data. You must supply licensed PT list.
"""
from __future__ import annotations
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def read_pt_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_name = row.get('pt_name', '').strip()
            pt_code = row.get('pt_code', '').strip()
            if not pt_name:
                continue
            mapping[pt_name] = pt_code
    return mapping

def read_synonyms(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            syn = row.get('synonym', '').strip()
            pt_name = row.get('pt_name', '').strip()
            if syn and pt_name:
                pairs.append((syn, pt_name))
    return pairs

def build_rows(pt_map: Dict[str, str], syn_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    # Add PTs themselves as canonical terms
    for pt_name, pt_code in pt_map.items():
        rows.append((pt_name, pt_name, 'meddra_pt', pt_code))
    # Add synonyms referencing PT
    for syn, pt_name in syn_pairs:
        pt_code = pt_map.get(pt_name, '')
        rows.append((syn, pt_name, 'user_syn', pt_code))
    # Deduplicate by (term, canonical)
    dedup: Dict[Tuple[str, str], Tuple[str, str, str, str]] = {}
    for term, canonical, source, code in rows:
        key = (term.lower(), canonical.lower())
        if key not in dedup:
            dedup[key] = (term, canonical, source, code)
        else:
            # Prefer row with a code present
            existing = dedup[key]
            if not existing[3] and code:
                dedup[key] = (term, canonical, source, code)
    return list(dedup.values())

def write_output(rows: List[Tuple[str, str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['term','canonical','source','concept_id'])
        for term, canonical, source, code in sorted(rows, key=lambda r: (r[1].lower(), r[0].lower())):
            writer.writerow([term, canonical, source, code])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt-file', required=True, type=Path, help='CSV with MedDRA PT list (pt_name, pt_code)')
    ap.add_argument('--syn-file', required=True, type=Path, help='CSV with synonym mappings (synonym, pt_name)')
    ap.add_argument('--out-file', required=True, type=Path, help='Destination CSV lexicon path')
    args = ap.parse_args()

    if not args.pt_file.exists():
        raise SystemExit(f"PT file not found: {args.pt_file}")
    if not args.syn_file.exists():
        raise SystemExit(f"Synonym file not found: {args.syn_file}")

    pt_map = read_pt_file(args.pt_file)
    if not pt_map:
        raise SystemExit("PT file contained no rows.")
    syn_pairs = read_synonyms(args.syn_file)
    if not syn_pairs:
        print("Warning: synonym file empty; only PT rows will be emitted.")

    rows = build_rows(pt_map, syn_pairs)
    write_output(rows, args.out_file)
    print(f"Wrote {len(rows)} rows -> {args.out_file}")

if __name__ == '__main__':
    main()
