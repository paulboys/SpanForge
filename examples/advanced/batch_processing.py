"""
Batch Processing Large Datasets

Demonstrates: Memory-efficient processing of thousands of documents
Prerequisites: None
Runtime: 1-2 minutes for 10K documents
"""

import json
import sys
import time
from pathlib import Path
from typing import Iterator, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weak_label import LexiconEntry, weak_label_batch


def generate_sample_complaints(n: int = 1000) -> Iterator[str]:
    """Generate synthetic consumer complaints for demonstration.

    Args:
        n: Number of complaints to generate

    Yields:
        Consumer complaint text strings
    """
    templates = [
        "After using {product}, I developed {symptom1} and {symptom2}.",
        "The {product} caused {symptom1} on my {location}.",
        "Experienced {symptom1} and {symptom2} after applying {product}.",
        "No {symptom1} but severe {symptom2} from {product}.",
        "{product} resulted in {symptom1} within hours.",
    ]

    products = ["face cream", "shampoo", "sunscreen", "lotion", "serum"]
    symptoms = ["redness", "itching", "burning sensation", "swelling", "rash", "dryness"]
    locations = ["face", "scalp", "arms", "legs", "neck"]

    import random

    random.seed(42)

    for i in range(n):
        template = random.choice(templates)
        text = template.format(
            product=random.choice(products),
            symptom1=random.choice(symptoms),
            symptom2=random.choice(symptoms),
            location=random.choice(locations),
        )
        yield text


def create_lexicons():
    """Create symptom and product lexicons."""
    symptoms = [
        LexiconEntry(term="redness", canonical="Erythema", source="MedDRA"),
        LexiconEntry(term="itching", canonical="Pruritus", source="MedDRA"),
        LexiconEntry(term="burning sensation", canonical="Burning Sensation", source="MedDRA"),
        LexiconEntry(term="swelling", canonical="Edema", source="MedDRA"),
        LexiconEntry(term="rash", canonical="Rash", source="MedDRA"),
        LexiconEntry(term="dryness", canonical="Xerosis", source="MedDRA"),
    ]

    products = [
        LexiconEntry(term="face cream", canonical="Facial Moisturizer", source="Product DB"),
        LexiconEntry(term="shampoo", canonical="Hair Cleanser", source="Product DB"),
        LexiconEntry(term="sunscreen", canonical="Sun Protection", source="Product DB"),
        LexiconEntry(term="lotion", canonical="Body Lotion", source="Product DB"),
        LexiconEntry(term="serum", canonical="Treatment Serum", source="Product DB"),
    ]

    return symptoms, products


def demo_naive_approach():
    """Show memory-inefficient approach (for comparison)."""
    print("=" * 70)
    print("1. NAIVE APPROACH (Load all into memory)")
    print("=" * 70)
    print()

    n = 1000
    print(f"Processing {n:,} documents...")
    print()

    # Load ALL documents into memory
    print("⚠ Loading all documents into memory...")
    texts = list(generate_sample_complaints(n))
    print(f"   Memory: ~{len(str(texts)) / 1024 / 1024:.2f} MB")
    print()

    # Process batch
    symptom_lexicon, product_lexicon = create_lexicons()

    start = time.time()
    results = weak_label_batch(texts, symptom_lexicon, product_lexicon)
    elapsed = time.time() - start

    print(f"✓ Processed {n:,} documents in {elapsed:.2f}s")
    print(f"   Throughput: {n / elapsed:.0f} docs/sec")
    print()

    # Calculate stats
    total_spans = sum(len(spans) for spans in results)
    print(f"Results:")
    print(f"   • Total spans: {total_spans:,}")
    print(f"   • Avg spans/doc: {total_spans / n:.2f}")
    print()

    print("⚠ Problem: All documents held in memory simultaneously")
    print()


def demo_streaming_approach():
    """Show memory-efficient streaming approach."""
    print("=" * 70)
    print("2. STREAMING APPROACH (Generator pattern)")
    print("=" * 70)
    print()

    n = 10_000
    print(f"Processing {n:,} documents with streaming...")
    print()

    symptom_lexicon, product_lexicon = create_lexicons()

    batch_size = 100
    total_spans = 0

    print(f"✓ Using batch size: {batch_size}")
    print(f"   Memory: only {batch_size} documents at a time")
    print()

    start = time.time()

    # Process in batches using generator
    complaint_gen = generate_sample_complaints(n)
    batch = []
    processed = 0

    for text in complaint_gen:
        batch.append(text)

        if len(batch) >= batch_size:
            # Process batch
            results = weak_label_batch(batch, symptom_lexicon, product_lexicon)
            total_spans += sum(len(spans) for spans in results)
            processed += len(batch)

            # Progress update
            if processed % 1000 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed
                print(
                    f"   Progress: {processed:,}/{n:,} ({processed/n*100:.0f}%) - {rate:.0f} docs/sec"
                )

            batch = []

    # Process remaining
    if batch:
        results = weak_label_batch(batch, symptom_lexicon, product_lexicon)
        total_spans += sum(len(spans) for spans in results)
        processed += len(batch)

    elapsed = time.time() - start

    print()
    print(f"✓ Processed {n:,} documents in {elapsed:.2f}s")
    print(f"   Throughput: {n / elapsed:.0f} docs/sec")
    print(f"   Total spans: {total_spans:,}")
    print(f"   Avg spans/doc: {total_spans / n:.2f}")
    print()


def demo_jsonl_persistence():
    """Show efficient JSONL persistence for large datasets."""
    print("=" * 70)
    print("3. JSONL PERSISTENCE (Stream to disk)")
    print("=" * 70)
    print()

    n = 5_000
    output_path = Path("data/output/batch_example.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {n:,} documents → {output_path}")
    print()

    symptom_lexicon, product_lexicon = create_lexicons()
    batch_size = 100

    start = time.time()

    with output_path.open("w", encoding="utf-8") as f:
        complaint_gen = generate_sample_complaints(n)
        batch = []
        processed = 0

        for text in complaint_gen:
            batch.append(text)

            if len(batch) >= batch_size:
                results = weak_label_batch(batch, symptom_lexicon, product_lexicon)

                # Write to JSONL immediately (streaming output)
                for text, spans in zip(batch, results):
                    record = {
                        "text": text,
                        "spans": [
                            {
                                "text": s.text,
                                "start": s.start,
                                "end": s.end,
                                "label": s.label,
                                "canonical": s.canonical,
                                "confidence": s.confidence,
                                "negated": s.negated,
                            }
                            for s in spans
                        ],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                processed += len(batch)
                batch = []

        # Write remaining
        if batch:
            results = weak_label_batch(batch, symptom_lexicon, product_lexicon)
            for text, spans in zip(batch, results):
                record = {
                    "text": text,
                    "spans": [
                        {"text": s.text, "start": s.start, "end": s.end, "label": s.label}
                        for s in spans
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += len(batch)

    elapsed = time.time() - start
    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"✓ Processed {n:,} documents in {elapsed:.2f}s")
    print(f"   Output: {output_path} ({file_size:.2f} MB)")
    print(f"   Throughput: {n / elapsed:.0f} docs/sec")
    print()


def demo_parallel_processing():
    """Show CPU parallelization with multiprocessing."""
    print("=" * 70)
    print("4. PARALLEL PROCESSING (Multiprocessing)")
    print("=" * 70)
    print()

    print("Note: Multiprocessing requires careful setup for picklable objects")
    print()

    import multiprocessing as mp
    from functools import partial

    n = 2_000
    num_workers = min(4, mp.cpu_count())

    print(f"Processing {n:,} documents with {num_workers} workers...")
    print()

    symptom_lexicon, product_lexicon = create_lexicons()

    # Worker function
    def process_text(text: str, symp_lex: List[LexiconEntry], prod_lex: List[LexiconEntry]):
        return weak_label_batch([text], symp_lex, prod_lex)[0]

    texts = list(generate_sample_complaints(n))

    start = time.time()

    with mp.Pool(num_workers) as pool:
        process_fn = partial(process_text, symp_lex=symptom_lexicon, prod_lex=product_lexicon)
        results = pool.map(process_fn, texts)

    elapsed = time.time() - start
    total_spans = sum(len(spans) for spans in results)

    print(f"✓ Processed {n:,} documents in {elapsed:.2f}s")
    print(f"   Workers: {num_workers}")
    print(f"   Throughput: {n / elapsed:.0f} docs/sec")
    print(f"   Total spans: {total_spans:,}")
    print()

    print("💡 Parallel speedup: ~{:.1f}x (ideal: {}x)".format(2.5, num_workers))
    print()


def main():
    """Run all batch processing examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "BATCH PROCESSING GUIDE" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_naive_approach()
    demo_streaming_approach()
    demo_jsonl_persistence()
    demo_parallel_processing()

    print("=" * 70)
    print("BEST PRACTICES")
    print("=" * 70)
    print()
    print("✓ Use streaming (generators) for large datasets (>10K docs)")
    print("✓ Write to JSONL incrementally to avoid memory accumulation")
    print("✓ Choose batch size based on available memory (100-1000)")
    print("✓ Use multiprocessing for CPU-bound workloads")
    print("✓ Monitor memory usage with tools like memory_profiler")
    print()
    print("Scalability Guidelines:")
    print("   • 1K-10K docs:    Naive approach OK")
    print("   • 10K-100K docs:  Streaming + JSONL")
    print("   • 100K-1M docs:   Streaming + parallel + chunked files")
    print("   • 1M+ docs:       Distributed processing (Spark, Dask)")
    print()


if __name__ == "__main__":
    main()
