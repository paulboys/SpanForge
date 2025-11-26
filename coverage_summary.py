import json

with open('coverage.json') as f:
    data = json.load(f)

files = data['files']
sorted_files = sorted(files.items(), key=lambda x: x[1]['summary']['percent_covered'])

print("\nCoverage by file (sorted by coverage %):\n")
print(f"{'File':<60} {'Coverage':>10} {'Missing':>10}")
print("=" * 82)

for filepath, info in sorted_files:
    percent = info['summary']['percent_covered']
    missing = info['summary']['missing_lines']
    print(f"{filepath:<60} {percent:>9.1f}% {missing:>10}")

print("\n" + "=" * 82)
total = data['totals']
print(f"{'TOTAL':<60} {total['percent_covered']:>9.1f}% {total['missing_lines']:>10}")
