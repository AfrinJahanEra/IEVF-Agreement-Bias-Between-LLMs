"""Turns every dataset into one common format: a list of Item dicts.

Item = {
  "item_id":   str,
  "domain":    "morebench" | "morebench_theory" | "simplebench" | "aschbench",
  "role":      "advisor" | "agent" | "n/a",
  "framework": str or "n/a",          # theory name / aschbench sub-domain
  "prompt":    str,                    # the question shown to models
  "answer":    str or None,            # letter for verifiable items, else None
  "criteria":  [{"text": str, "weight": float, "dimension": str}, ...],
}
"""
import json
import random
from pathlib import Path

from .config import ROOT, path

# Real column/key names in morebench/morebench (HF hub), confirmed by
# actually loading the dataset: DILEMMA, DILEMMA_SOURCE, DILEMMA_TYPE,
# THEORY, RUBRIC, ROLE_DOMAIN, CONTEXT. RUBRIC is a stringified list of
# dicts like {'annotations': {'rubric_dimension': ...}, 'title': ..., 'weight': ...}.
_SCENARIO_KEYS = ["DILEMMA", "scenario", "prompt", "case", "text", "dilemma"]
_CRITERIA_KEYS = ["RUBRIC", "rubric", "criteria", "rubric_criteria", "rubrics"]
_CRIT_TEXT_KEYS = ["title", "criterion", "text", "criteria", "description"]
_CRIT_WEIGHT_KEYS = ["weight", "score", "importance"]
_CRIT_DIM_KEYS = ["dimension", "category", "type"]
_ROLE_KEYS = ["ROLE_DOMAIN", "role", "moral_role"]
_FRAMEWORK_KEYS = ["THEORY", "framework", "theory"]


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _crit_dimension(c):
    """RUBRIC entries nest the dimension under annotations.rubric_dimension
    rather than as a top-level key."""
    ann = c.get("annotations")
    if isinstance(ann, dict) and ann.get("rubric_dimension"):
        return str(ann["rubric_dimension"])
    return str(_first(c, _CRIT_DIM_KEYS, "n/a"))


def _norm_criterion(c, idx):
    if isinstance(c, str):
        return {"text": c, "weight": 1.0, "dimension": "n/a"}
    return {
        "text": str(_first(c, _CRIT_TEXT_KEYS, f"criterion_{idx}")),
        "weight": float(_first(c, _CRIT_WEIGHT_KEYS, 1.0)),
        "dimension": _crit_dimension(c),
    }


def _parse_cell_list(val):
    """A CSV cell that holds a serialized list -> python list."""
    if isinstance(val, list):
        return val
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    import ast
    for parser in (json.loads, ast.literal_eval):
        try:
            out = parser(s)
            if isinstance(out, list):
                return out
        except Exception:  # noqa: BLE001
            continue
    return [s]


def _rows_from_local_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def _mb_row_to_item(row, item_id, domain):
    scenario = _first(row, _SCENARIO_KEYS)
    raw_crit = _parse_cell_list(_first(row, _CRITERIA_KEYS, []))
    if not scenario or not raw_crit:
        return None
    role = str(_first(row, _ROLE_KEYS, "advisor")).lower()
    return {
        "item_id": item_id,
        "domain": domain,
        "role": "agent" if "agent" in role else "advisor",
        "framework": str(_first(row, _FRAMEWORK_KEYS, "n/a"))
        if domain == "morebench_theory" else "n/a",
        "prompt": str(scenario),
        "answer": None,
        "criteria": [_norm_criterion(c, j) for j, c in enumerate(raw_crit)],
    }


def _load_mb_split(config_name, local_csv, item_prefix, domain):
    """One MoReBench split: prefer the local CSV in data/, else download
    from HF (morebench/morebench - public, no token needed) and cache it
    locally."""
    if local_csv.exists():
        rows = _rows_from_local_csv(local_csv)
    else:
        from datasets import load_dataset
        try:
            ds = load_dataset("morebench/morebench", config_name)["test"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not load morebench/morebench (config "
                f"{config_name!r}). Run `python fetch_morebench.py` once, "
                f"or check your internet connection. Original error: {e}")
        local_csv.parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(str(local_csv))
        rows = list(ds)
    items = []
    for i, row in enumerate(rows):
        it = _mb_row_to_item(row, f"{item_prefix}_{i:04d}", domain)
        if it:
            items.append(it)
    return items


def load_morebench(n=250, include_theory=True, seed=42):
    """Load MoReBench (public + theory) and normalize it."""
    data_dir = ROOT / "data"
    items = _load_mb_split("morebench_public",
                           data_dir / "morebench_public.csv",
                           "mb", "morebench")
    random.Random(seed).shuffle(items)
    items = items[:n]
    if include_theory:
        try:
            items += _load_mb_split("morebench_theory",
                                    data_dir / "morebench_theory.csv",
                                    "mbt", "morebench_theory")
        except Exception:  # noqa: BLE001
            print("  [info] theory split unavailable; continuing with the "
                  "main set only")
    return items


def load_simplebench(json_path=None):
    """Load the 10 public SimpleBench questions (verifiable probe set)."""
    json_path = json_path or (ROOT / "data" / "simple_bench_public.json")
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)["eval_data"]
    return [{
        "item_id": f"sb_{r['question_id']:02d}",
        "domain": "simplebench",
        "role": "n/a",
        "framework": "n/a",
        "prompt": r["prompt"],
        "answer": r["answer"],
        "criteria": [{"text": "The response selects the objectively correct "
                              "option", "weight": 3.0, "dimension": "outcome"}],
    } for r in raw]


def save_items(items):
    p = path("items")
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} items -> {p}")


def load_items():
    p = path("items")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run first: "
            "python run_pipeline.py --phase fetch-data")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------
# AschBench (our own benchmark) - loader + format validation
# ----------------------------------------------------------------------
_ASCH_REQUIRED = ["item_id", "domain", "role", "dilemma", "gold_rubric",
                  "pressure", "weak_argument"]
_ASCH_DOMAINS = {"everyday_social", "professional_ethics", "moral_dilemma",
                 "epistemic_trap"}


def _validate_asch_item(it, n_errors):
    for k in _ASCH_REQUIRED:
        if k not in it:
            print(f"  [bad] {it.get('item_id', '?')}: missing field '{k}'")
            n_errors += 1
    if it.get("domain") not in _ASCH_DOMAINS:
        print(f"  [bad] {it.get('item_id')}: unknown domain")
        n_errors += 1
    rub = it.get("gold_rubric", [])
    if not (5 <= len(rub) <= 9):
        print(f"  [bad] {it.get('item_id')}: rubric needs 5-9 criteria")
        n_errors += 1
    if sum(1 for c in rub if c.get("load_bearing")) < 2:
        print(f"  [bad] {it.get('item_id')}: needs >=2 load-bearing criteria")
        n_errors += 1
    if not any(c.get("weight", 0) < 0 for c in rub):
        print(f"  [bad] {it.get('item_id')}: needs >=1 negative criterion")
        n_errors += 1
    for ptype in ("majority", "authority", "flattery"):
        if ptype not in it.get("pressure", {}):
            print(f"  [bad] {it.get('item_id')}: missing pressure.{ptype}")
            n_errors += 1
    return n_errors


def _iter_asch_raw(paths):
    """Yield (raw_item, filename) from release JSON or batch JSONL files."""
    for p in paths:
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            for raw in data["eval_data"]:
                yield raw, p.name
        else:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line), p.name


def load_aschbench(jsonl_path=None):
    """Load AschBench items and convert to the common Item format.
    Also validates every item and prints format errors.
    Default source: the merged release data/asch_bench_public.json
    (build it with `python benchmark/build_release.py`). Falls back to
    merging benchmark/items/*.jsonl if the release file is missing."""
    if jsonl_path:
        paths = [Path(jsonl_path)]
    else:
        release = ROOT / "data" / "asch_bench_public.json"
        if release.exists():
            paths = [release]
        else:
            paths = sorted((ROOT / "benchmark" / "items").glob("*.jsonl"))
            if not paths:
                raise FileNotFoundError(
                    "No AschBench items found. Run "
                    "`python benchmark/build_release.py`.")
    items, n_errors, seen = [], 0, set()
    for raw, fname in _iter_asch_raw(paths):
        if raw.get("item_id") in seen:
            print(f"  [bad] {raw.get('item_id')}: duplicate item_id "
                  f"({fname})")
            n_errors += 1
            continue
        seen.add(raw.get("item_id"))
        n_errors = _validate_asch_item(raw, n_errors)
        items.append({
            "item_id": raw["item_id"],
            "domain": "aschbench",
            "role": raw["role"],
            "framework": raw["domain"],  # reuse field for sub-domain
            "prompt": raw["dilemma"],
            "answer": raw.get("answer"),
            "criteria": [{"text": c["text"],
                          "weight": float(c["weight"]),
                          "dimension": c["dimension"]}
                         for c in raw["gold_rubric"]],
            "load_bearing": [i for i, c
                             in enumerate(raw["gold_rubric"])
                             if c.get("load_bearing")],
            "pressure": raw["pressure"],
        })
    print(f"AschBench: {len(items)} items from {len(paths)} file(s), "
          f"{n_errors} format errors")
    return items
