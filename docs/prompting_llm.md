# LLM-Assisted Refinement (Experimental)

SpanForge optionally applies Large Language Model (LLM) prompts to improve weak label span boundaries, negation status, and canonical mapping prior to human annotation.

## Goals
- Tighten span boundaries and reduce annotator edit time.
- Identify ambiguous or mid-confidence spans for targeted refinement.
- Provide provenance and reproducibility (prompt version, model, provider).

## Workflow Insertion
```
Raw Text → Weak Labeler → (OPTIONAL) LLM Refinement → Label Studio Annotation → Gold Conversion → Quality / Registry
```

## Configuration (src/config.py)
Fields:
- `llm_enabled`: Master toggle (default false).
- `llm_provider`: Stub/openai/etc.
- `llm_model`: Model identifier (e.g. gpt-4).
- `llm_min_confidence`: Minimum confidence to keep LLM suggestion.
- `llm_cache_path`: Future caching location.
- `llm_prompt_version`: Version tag for prompt templates.

## Prompts
Templates stored under `prompts/`:
- `boundary_refine.txt`: Adjust span boundaries.
- `negation_check.txt`: Negation classification.
- `synonym_expand.txt`: Canonical + synonyms.

Templates include placeholders: `{{text}}`, `{{candidates}}`, `{{knowledge}}`.

## Provenance Fields
Each suggestion appended in `refine_llm.py` includes:
- `source = llm_refine`
- `llm_confidence`
- `confidence_reason` (future, model-supplied)
- `canonical` (if provided)
- `prompt_version`, `model`, `provider` in `llm_meta`

## CLI Usage
```
python scripts/annotation/cli.py refine-llm \
  --weak data/output/weak.jsonl \
  --out data/output/refined_weak.jsonl \
  --prompt prompts/boundary_refine.txt
```
Add `--dry-run` to preview without writing output.

## Filtering Logic
Only spans with heuristic confidence in mid band (0.55–0.75) are candidates.

## Stub Behavior
Current implementation returns no suggestions (stub). Integrate provider later by extending `LLMAgent.call()`.

## Extending to Real Provider
1. Inject API key via environment variable.
2. Replace stub call with provider client invocation.
3. Enforce `temperature=0` and JSON schema.
4. Implement retry + minimal exponential backoff.

## Evaluation (Future)
Script: `evaluate_refinement.py` (planned) will measure:
- Boundary IOU uplift versus original weak spans.
- Conflict reduction after adjudication.
- Long-tail canonical discovery.

## Safety & Privacy
- De-identify raw complaint text before external calls.
- Log prompt hashes for reproducibility.
- Maintain local cache to minimize repeated transmissions.

## Next Steps
- Implement real provider integration.
- Add evaluation script and tests for parsing.
- Incorporate acceptance metrics in registry.
