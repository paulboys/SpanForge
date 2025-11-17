<div align="center">
	<img src="assets/SpanForge.png" alt="SpanForge Logo" width="140" />
	<h2 style="margin-top:0">SpanForge Annotation Guide</h2>
	<p style="max-width:560px">Standards for consistent SYMPTOM and PRODUCT span curation enabling high-fidelity adverse event modeling.</p>
	<hr style="width:55%;border:0;border-top:1px solid #ddd" />
</div>

# Annotation Guide

Defines consistent rules for annotating SYMPTOM and PRODUCT spans in consumer adverse event complaints.

## Objectives
1. Capture clinically meaningful symptom phrases (granular, complete).
2. Standardize product references for model association.
3. Preserve negated context (annotate + flag) for future modeling of absence.
4. Minimize ambiguity and overlapping conflicts.

## Labels
- **SYMPTOM**: Physiological or subjective adverse effect ("redness", "severe rash", "nausea", "stinging").
- **PRODUCT**: Product name/formulation or clear category reference ("vitamin serum", "exfoliating scrub").

## Span Boundary Rules
| Rule | Examples |
|------|----------|
| Include modifiers integral to meaning | `severe rash`, `mild dryness`, `burning pain` |
| Exclude trailing punctuation | `itching.` → `itching` |
| Avoid partial capture | Prefer `tiny itching spots` over `itching` alone |
| Keep internal spacing & casing as-is | `hydra boost cream` preserved |
| Exclude unrelated conjunctions | `rash and` → `rash` |

## Negation Handling (Annotate + Flag)
Annotate negated symptoms ("no irritation", "without redness") and rely on conversion phase to set `negated=True`. This supports training for presence vs absence.

Do NOT annotate if term clearly unrelated to adverse context (e.g., "no product issues" → skip `issues`).

## Anatomy Tokens
Skip isolated anatomy (`face`, `skin`) unless part of explicit symptom phrase ("skin irritation" → annotate `skin irritation`).

## Overlaps & Nested Spans
- Choose the most semantically complete span (`severe burning pain` preferred).
- If two plausible alternatives and uncertainty persists: keep both → adjudication tool resolves.

## Product vs Symptom Separation
If a product term appears inside a symptom phrase but functions as a product reference, separate spans where boundaries are clean: `serum-induced itching` → `serum` (PRODUCT), `itching` (SYMPTOM).

## Canonical Mapping
Surface span text is later mapped to canonical lexicon entries; missing variants should be added to lexicon CSVs. Do not force canonical wording during annotation—capture verbatim text.

## Conflict Resolution (Planned Consensus)
1. Exact match majority for identical spans.
2. Longest span tie-breaker when semantics equivalent.
3. Differing labels on overlap → adjudication review output to `data/annotation/conflicts/`.

## Provenance Fields (After Conversion)
`source`, `annotator`, `revision`, `canonical`, optional `concept_id` automatically injected—no manual action required in UI.

## Quality Checklist Before Export
- Boundaries precise (no punctuation, correct modifiers).
- Negated spans present (not deleted unless irrelevant context).
- No duplicate (start,end,label) tuples.
- Low conflict count (<5% tasks flagged).

## Edge Case Decisions
| Scenario | Action |
|----------|--------|
| "dry" vs "dryness" | Annotate verbatim form present |
| Misspelling ("nausia") | Annotate misspelling; canonical normalizes |
| Compound ("rash and itching") | Two spans if distinct sensations |
| Intensifier only ("very") | Exclude unless integral ("very dry skin" → include `very dry skin`) |
| Slang ("tummy pain") | Annotate; canonical maps to `abdominal pain` if lexicon contains mapping |

## Common Pitfalls
| Pitfall | Correction |
|---------|------------|
| Dropping severity adjective | Include full phrase |
| Deleting negated symptom | Keep & rely on negation flag |
| Including trailing period | Trim punctuation |
| Over-extending into next clause | Limit to symptom/product phrase only |

## Updating This Guide
Revise after initial adjudication cycle (≈ first 100 gold tasks). All changes recorded in provenance registry notes for traceability.

## FAQ
**Should I annotate brand names?** Yes, if they directly relate to the adverse context.

**Annotate plural symptoms?** Yes; canonical mapping handles singular normalization.

**What about uncertain reactions?** Annotate if consumer asserts possibility ("might be causing redness"). Model can later learn uncertainty patterns.

**Do I merge adjacent symptoms?** Only if forming a unified phrase ("redness and itching" → two spans).

## Next Steps
After annotation export: run conversion → quality report → register batch → prepare for BIO tagging.
