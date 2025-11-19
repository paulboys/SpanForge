# Placeholder lexicon builder.
# Intended flow:
# 1. Load source term lists (MedDRA export, CHV list, proprietary symptom terms)
# 2. Normalize casing, strip punctuation, deduplicate.
# 3. Map to canonical form + concept id when available.
# 4. Write to data/lexicon/symptoms.csv (term, canonical, source, concept_id).

import csv
from pathlib import Path

OUTPUT_DIR = Path("data/lexicon")
SYMPTOM_FILE = OUTPUT_DIR / "symptoms.csv"
PRODUCT_FILE = OUTPUT_DIR / "products.csv"

SAMPLE_SYMPTOMS = [
    {"term": "headache", "canonical": "headache", "source": "sample", "concept_id": "C0018681"},
    {"term": "skin rash", "canonical": "skin rash", "source": "sample", "concept_id": "C0037286"},
]

SAMPLE_PRODUCTS = [
    {"term": "moisturizing cream", "canonical": "Hydra Boost Moisturizing Cream", "sku": "SKU-1001", "category": "skincare"},
    {"term": "face wash", "canonical": "Gentle Daily Cleanser", "sku": "SKU-1002", "category": "skincare"},
    {"term": "serum", "canonical": "Radiance Vitamin C Serum", "sku": "SKU-1003", "category": "skincare"},
]

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def write_sample():
    ensure_output_dir()
    # Write symptoms
    with SYMPTOM_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "canonical", "source", "concept_id"])
        writer.writeheader()
        for row in SAMPLE_SYMPTOMS:
            writer.writerow(row)
    print(f"Wrote sample symptom lexicon to {SYMPTOM_FILE}")
    
    # Write products
    with PRODUCT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "canonical", "sku", "category"])
        writer.writeheader()
        for row in SAMPLE_PRODUCTS:
            writer.writerow(row)
    print(f"Wrote sample product lexicon to {PRODUCT_FILE}")

if __name__ == "__main__":
    write_sample()
