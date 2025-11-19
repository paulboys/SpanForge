#!/usr/bin/env python
"""Minimal training scaffold for token classification.

NOT production-ready: focuses on transforming gold JSONL spans into BIO tags and running a simple training loop.

Usage (example):
  python scripts/train_token_cls.py \
    --gold data/annotation/exports/gold_full.jsonl \
    --labels labels.json \
    --output models/token_cls_trial

Assumptions:
  - Gold JSONL lines contain: {id, text, entities:[{start,end,label}]}
  - Labels.json defines BIO tag inventory.
  - Sequence truncation at config.max_seq_len.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW

from src.config import get_config
from src.model_token_cls import get_token_cls_model, load_labels


@dataclass
class Record:
    rid: str
    text: str
    entities: List[Dict[str, Any]]


def read_gold(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(Record(rid=str(obj.get("id")), text=obj.get("text", ""), entities=obj.get("entities", [])))
    return records


def spans_to_bio(tokens: List[str], offsets: List[tuple], entities: List[Dict[str, Any]], label2id: Dict[str, int]) -> List[int]:
    # Build char-level map to entity label
    tags = ["O"] * len(tokens)
    for ent in entities:
        label = ent.get("label")
        start = ent.get("start")
        end = ent.get("end")
        if label is None or start is None or end is None:
            continue
        for i, (s, e) in enumerate(offsets):
            if e <= start or s >= end:
                continue
            if s >= start and e <= end:  # token fully inside span
                prefix = "B" if tags[i] == "O" else "I"
                base = label
                tags[i] = f"{prefix}-{base}"
    # Convert to ids (fallback to O)
    return [label2id.get(t, label2id["O"]) for t in tags]


class TokenClsDataset(Dataset):
    def __init__(self, records: List[Record], tokenizer, label2id: Dict[str, int], max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        encoding = self.tokenizer(r.text, return_offsets_mapping=True, truncation=True, max_length=self.max_len)
        offsets = encoding.pop("offset_mapping")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        labels = spans_to_bio(tokens, offsets, r.entities, self.label2id)
        # Pad labels if shorter due to truncation or special tokens
        if len(labels) < len(input_ids):
            labels.extend([self.label2id["O"]] * (len(input_ids) - len(labels)))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def train(model, dataloader, epochs: int, lr: float, device: str, output_dir: Path):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} - loss: {avg:.4f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train token classification head (scaffold)")
    parser.add_argument("--gold", required=True, help="Gold JSONL file path")
    parser.add_argument("--labels", default="labels.json", help="Labels JSON file")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    config = get_config()
    labels = load_labels(Path(args.labels))
    _, model = get_token_cls_model(Path(args.labels))
    label2id = {l: i for i, l in enumerate(labels)}
    tokenizer, _ = get_token_cls_model(Path(args.labels))

    records = read_gold(Path(args.gold))
    if not records:
        raise SystemExit("No gold records found.")
    dataset = TokenClsDataset(records, tokenizer, label2id, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train(model, dataloader, args.epochs, args.lr, config.device, Path(args.output))


if __name__ == "__main__":
    main()
