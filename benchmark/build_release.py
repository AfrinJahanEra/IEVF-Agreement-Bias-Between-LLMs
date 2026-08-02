"""Merge all benchmark/items/*.jsonl batches into ONE release file:

    data/asch_bench_public.json   {"eval_data": [ {...}, ... ]}

Same envelope style and same folder as data/simple_bench_public.json,
but every entry keeps the full AschBench fields (gold_rubric, pressure,
weak_argument, ...).

Usage (from the project root):
    python benchmark/build_release.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `src` is importable

from src.data_loader import _validate_asch_item  # noqa: E402

ITEMS_DIR = ROOT / "benchmark" / "items"
OUT = ROOT / "data" / "asch_bench_public.json"


def main():
    paths = sorted(ITEMS_DIR.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No item batches in {ITEMS_DIR}")
    items, seen, n_errors = [], set(), 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("item_id") in seen:
                    print(f"  [bad] duplicate item_id: {raw.get('item_id')}"
                          f" ({p.name})")
                    n_errors += 1
                    continue
                seen.add(raw.get("item_id"))
                n_errors = _validate_asch_item(raw, n_errors)
                items.append(raw)
    if n_errors:
        print(f"ABORT: {n_errors} validation errors - fix before release.")
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"eval_data": items}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"OK: {len(items)} items from {len(paths)} batches -> {OUT}")


if __name__ == "__main__":
    main()
