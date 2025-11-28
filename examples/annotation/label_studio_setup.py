"""
Annotation Example: Label Studio Setup
=======================================

Demonstrates how to configure Label Studio for biomedical NER annotation,
including project setup, label configuration, hotkeys, and best practices
for annotator workflows.

**What You'll Learn:**
- Creating Label Studio projects programmatically
- Configuring SYMPTOM/PRODUCT labels with hotkeys
- Setting up annotation guidelines
- Importing pre-annotated weak labels
- Managing annotator accounts and permissions

**Prerequisites:**
- Label Studio installed (pip install label-studio)
- Completed annotation/prepare_batch.py
- Understanding of annotation workflows

**Runtime:** ~2 minutes (includes Label Studio API calls)

**Use Cases:**
- Setting up annotation project for first time
- Onboarding new annotators
- Configuring quality control workflows
- Managing multi-annotator projects
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Note: This example shows the configuration patterns
# Actual Label Studio integration requires the label-studio package


def demo_label_config():
    """Show the Label Studio XML configuration."""
    print("\n" + "=" * 70)
    print("⚙️  LABEL STUDIO CONFIGURATION")
    print("=" * 70)

    label_config = """<View>
  <Header value="Biomedical NER: Symptom and Product Extraction"/>
  
  <Text name="text" value="$text" granularity="word" highlightColor="#09a3d5"/>
  
  <Labels name="entities" toName="text">
    <Label value="SYMPTOM" background="#FFA07A" hotkey="s"/>
    <Label value="PRODUCT" background="#98D8C8" hotkey="p"/>
  </Labels>
  
  <Header value="Instructions:"/>
  <View style="margin-top: 10px; padding: 10px; background: #f5f5f5;">
    <Text value="1. Select text by clicking and dragging"/>
    <Text value="2. Press 's' for SYMPTOM or 'p' for PRODUCT"/>
    <Text value="3. Include full medical phrase (e.g., 'burning sensation')"/>
    <Text value="4. Exclude trailing punctuation"/>
    <Text value="5. Annotate negated symptoms (e.g., 'no redness')"/>
  </View>
</View>"""

    print("\n📄 Label Configuration (XML):")
    print(label_config)

    print("\n✨ Configuration Features:")
    print("   • Word-level granularity (prevents mid-word selection)")
    print("   • Hotkeys: 's' for SYMPTOM, 'p' for PRODUCT")
    print("   • Colorblind-safe palette (orange/teal)")
    print("   • Built-in instructions visible to annotators")
    print("   • Header for context")

    print("\n💡 To Use:")
    print("   1. Copy the XML above")
    print("   2. In Label Studio: Settings → Labeling Interface")
    print("   3. Paste XML and save")
    print("   4. Preview with sample task")


def demo_project_setup():
    """Show how to set up a Label Studio project."""
    print("\n" + "=" * 70)
    print("📁 PROJECT SETUP")
    print("=" * 70)

    print("\n🔧 Project Configuration:\n")

    project_config = {
        "title": "SpanForge Biomedical NER",
        "description": "Annotate symptoms and products in consumer complaints",
        "label_config": "<See above XML>",
        "expert_instruction": (
            "Annotate all symptom and product mentions. "
            "Include full clinical phrases. Exclude adjectives like 'severe' or 'mild'. "
            "Mark negated symptoms for separate handling."
        ),
        "show_instruction": True,
        "show_skip_button": True,
        "show_submit_button": True,
        "show_ground_truth_first": True,
    }

    for key, value in project_config.items():
        print(f"   {key}:")
        if isinstance(value, str) and len(value) > 60:
            print(f"      {value[:60]}...")
        else:
            print(f"      {value}")

    print("\n📊 Recommended Settings:")
    print("   • Enable annotation review: Yes")
    print("   • Sampling: Sequential (for calibration batches)")
    print("   • Task overlap: 2 (for IAA calculation)")
    print("   • Skip queue: Disabled (ensure complete annotation)")

    print("\n💡 Manual Setup Steps:")
    print("   1. Launch Label Studio:")
    print("      label-studio start")
    print("   2. Create new project")
    print("   3. Paste label configuration")
    print("   4. Import tasks (JSONL format)")
    print("   5. Invite annotators")


def demo_task_import():
    """Show how to import tasks to Label Studio."""
    print("\n" + "=" * 70)
    print("📤 TASK IMPORT")
    print("=" * 70)

    print("\n📋 Import Format (JSONL):\n")

    sample_task = {
        "data": {
            "text": "After using the facial moisturizer, I developed burning sensation and redness."
        },
        "predictions": [
            {
                "model_version": "weak_labels_v1",
                "result": [
                    {
                        "value": {
                            "start": 40,
                            "end": 57,
                            "text": "burning sensation",
                            "labels": ["SYMPTOM"],
                        },
                        "from_name": "entities",
                        "to_name": "text",
                        "type": "labels",
                    },
                    {
                        "value": {"start": 62, "end": 69, "text": "redness", "labels": ["SYMPTOM"]},
                        "from_name": "entities",
                        "to_name": "text",
                        "type": "labels",
                    },
                ],
            }
        ],
    }

    import json

    print("   " + json.dumps(sample_task, indent=2).replace("\n", "\n   "))

    print("\n✨ Import Features:")
    print("   • 'data': Required - contains text to annotate")
    print("   • 'predictions': Optional - pre-annotations from weak labels")
    print("   • 'annotations': Optional - ground truth for gold examples")

    print("\n💡 Import Methods:")
    print("   1. Web UI: Project → Import → Upload JSONL")
    print("   2. Python SDK:")
    print("      from label_studio_sdk import Client")
    print("      ls = Client(url='http://localhost:8080', api_key='...')")
    print("      ls.import_tasks(project_id, 'batch_001.jsonl')")
    print("   3. REST API:")
    print("      curl -X POST http://localhost:8080/api/projects/{id}/import \\")
    print("           -H 'Authorization: Token ...' \\")
    print("           -F 'file=@batch_001.jsonl'")


def demo_annotation_guidelines():
    """Show annotation guidelines for annotators."""
    print("\n" + "=" * 70)
    print("📖 ANNOTATION GUIDELINES")
    print("=" * 70)

    print("\n🎯 General Principles:\n")

    guidelines = [
        (
            "Completeness",
            "Include full clinical phrase (e.g., 'burning sensation' not just 'burning')",
        ),
        ("Boundaries", "Exclude superfluous adjectives ('severe', 'mild', 'slight')"),
        ("Negation", "Annotate negated symptoms (e.g., 'no redness') - mark for separate handling"),
        ("Determiners", "Exclude articles ('the', 'a') unless part of product name"),
        ("Punctuation", "Exclude trailing punctuation (periods, commas)"),
        ("Ambiguity", "When uncertain, annotate the most specific span and flag for review"),
    ]

    for i, (principle, explanation) in enumerate(guidelines, 1):
        print(f"   {i}. {principle}:")
        print(f"      {explanation}\n")

    print("🏷️  Label Definitions:\n")

    labels = [
        (
            "SYMPTOM",
            "Medical symptoms, adverse effects, or reactions",
            "✓ burning, redness, swelling, itching, rash\n      ✗ face, skin (anatomy only)",
        ),
        (
            "PRODUCT",
            "Cosmetic products, medications, or supplements",
            "✓ face cream, moisturizer, serum, vitamin C\n      ✗ water, soap (if general reference)",
        ),
    ]

    for label, definition, examples in labels:
        print(f"   {label}:")
        print(f"      Definition: {definition}")
        print(f"      Examples:\n         {examples}\n")

    print("💡 Common Mistakes:\n")

    mistakes = [
        "❌ Including 'severe' in 'severe burning' → ✓ Just 'burning'",
        "❌ Annotating 'face' alone → ✓ Skip unless 'facial cream'",
        "❌ Missing negation 'no redness' → ✓ Annotate 'redness' + negation flag",
        "❌ Overlapping spans for same entity → ✓ Choose most complete span",
    ]

    for mistake in mistakes:
        print(f"   {mistake}")


def demo_quality_control():
    """Show quality control workflows."""
    print("\n" + "=" * 70)
    print("✅ QUALITY CONTROL")
    print("=" * 70)

    print("\n📊 Quality Assurance Workflow:\n")

    workflow = [
        (
            "Calibration Round",
            "First 20 tasks: All annotators work on same texts",
            "Calculate IAA (Cohen's kappa), discuss disagreements",
        ),
        (
            "Main Annotation",
            "Distribute tasks: 80% single annotator, 20% overlap",
            "Monitor annotation speed and agreement in real-time",
        ),
        (
            "Expert Review",
            "Review all low-agreement tasks (IOU < 0.7)",
            "Adjudicate conflicts, create consensus labels",
        ),
        (
            "Drift Detection",
            "Weekly: Check annotation consistency over time",
            "Re-train on gold examples if drift detected",
        ),
    ]

    for i, (phase, description, action) in enumerate(workflow, 1):
        print(f"   {i}. {phase}:")
        print(f"      Task: {description}")
        print(f"      Action: {action}\n")

    print("📈 Quality Metrics:\n")

    metrics = [
        (
            "IAA (Inter-Annotator Agreement)",
            "Cohen's kappa ≥ 0.75",
            "Measured on overlapping tasks",
        ),
        (
            "Annotation Speed",
            "5-10 tasks/hour",
            "Too fast → quality issue, too slow → complexity issue",
        ),
        ("Skip Rate", "< 5%", "High skip rate → unclear guidelines or difficult examples"),
        ("Revision Rate", "< 10%", "High revision → inconsistent annotators"),
    ]

    for metric, target, notes in metrics:
        print(f"   {metric}:")
        print(f"      Target: {target}")
        print(f"      Notes: {notes}\n")


def demo_export_workflow():
    """Show how to export annotated data."""
    print("\n" + "=" * 70)
    print("📥 EXPORT WORKFLOW")
    print("=" * 70)

    print("\n📤 Export Formats:\n")

    formats = [
        ("JSON", "Full annotation details with metadata", "Best for: Analysis, review, auditing"),
        ("JSONL", "One annotation per line", "Best for: Training, processing pipelines"),
        ("CSV", "Flat format with span positions", "Best for: Excel analysis, statistics"),
        ("COCO", "Computer vision format", "Not recommended for NER"),
    ]

    for format_name, description, use_case in formats:
        print(f"   {format_name}:")
        print(f"      {description}")
        print(f"      {use_case}\n")

    print("✅ Recommended Export:")
    print("   Format: JSONL")
    print("   Filter: Completed annotations only")
    print("   Include: annotator metadata, timestamps")
    print("   Exclude: predictions (keep only human annotations)")

    print("\n💡 Post-Export Processing:")
    print("   1. Convert to SpanForge gold standard format")
    print("      → See export_gold_standard.py")
    print("   2. Calculate inter-annotator agreement")
    print("      → See quality_report.py")
    print("   3. Evaluate against weak labels")
    print("      → See evaluation/compute_metrics.py")


def main():
    """Run all Label Studio setup examples."""
    print("\n" + "=" * 70)
    print("LABEL STUDIO SETUP EXAMPLES")
    print("=" * 70)
    print("\nDemonstrates configuring Label Studio for biomedical NER")
    print("annotation, including project setup, guidelines, quality control,")
    print("and export workflows.")

    # Run demos
    demo_label_config()
    demo_project_setup()
    demo_task_import()
    demo_annotation_guidelines()
    demo_quality_control()
    demo_export_workflow()

    print("\n" + "=" * 70)
    print("✓ All Label Studio setup examples completed!")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   • Install Label Studio: pip install label-studio")
    print("   • Launch server: label-studio start")
    print("   • Create project with configuration above")
    print("   • Import tasks from prepare_batch.py")
    print("   • Try export_gold_standard.py after annotation")
    print()


if __name__ == "__main__":
    main()
