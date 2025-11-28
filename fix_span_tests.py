"""Fix Span construction in test files to use keyword arguments."""

import re
from pathlib import Path


def fix_span_constructions(file_path: Path) -> int:
    """Fix all Span constructions to use keyword arguments.

    Pattern: Span("text", start, end, "label", "canonical", confidence, negated)
    Becomes: Span(text="text", start=start, end=end, label="label", canonical="canonical", confidence=confidence, negated=negated)
    """
    content = file_path.read_text(encoding="utf-8")

    # Pattern matches: Span("text", num, num, "LABEL", "Canonical", float, bool)
    pattern = r'Span\("([^"]+)",\s*(-?\d+),\s*(-?\d+),\s*"([^"]+)",\s*"([^"]*)",\s*([\d.]+),\s*(True|False)\)'
    replacement = (
        r'Span(text="\1", start=\2, end=\3, label="\4", canonical="\5", confidence=\6, negated=\7)'
    )

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"Fixed {count} Span constructions in {file_path.name}")
    else:
        print(f"No changes needed in {file_path.name}")

    return count


if __name__ == "__main__":
    test_dir = Path("tests/weak_labeling")
    total = 0

    for test_file in test_dir.glob("test_*.py"):
        total += fix_span_constructions(test_file)

    print(f"\nTotal: Fixed {total} Span constructions across all test files")
