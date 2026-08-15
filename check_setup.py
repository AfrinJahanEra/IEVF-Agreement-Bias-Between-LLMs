"""Preflight check - run before spending anything.

    python check_setup.py          # config, tokens, judges, data, database
    python check_setup.py --ping   # + one tiny REAL call per ready model

Nothing is faked here: --ping sends a 5-token question to each model whose
token is present, so a green line means that token really works.
"""
import argparse
import sqlite3

from src import models
from src.config import CFG, path


def show_models():
    print("--- models (config.yaml + .env) ---")
    ready = []
    for key in CFG["models"]:
        sp = models.spec(key)
        ok = models.has_token(key)
        mark = "OK  " if ok else "--  "
        token_name = sp.get("api_key_env") or "(none - runs locally)"
        local = " [local]" if models.is_local(key) else ""
        print(f"{mark}{key:<15} {sp['model']:<28} family={sp['family']:<10} "
              f"token={token_name}{local}{'' if ok else ' (missing)'}")
        if ok:
            ready.append(key)
    print("\n--- judges (independent grader per model) ---")
    for key in ready:
        try:
            print(f"OK  {key:<15} -> {models.judge_for(key)}")
        except SystemExit as e:
            print(f"!!  {key:<15} -> {e}")
    if not ready:
        print("No usable model: add a token to .env (see .env.example)")
    return ready


def show_data():
    print("\n--- question bank ---")
    p = path("items")
    if not p.exists():
        print(f"missing {p}: run "
              "python run_pipeline.py --phase fetch-data --simplebench "
              "--aschbench")
        return
    import json
    from collections import Counter
    with open(p, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"{len(items)} items: {dict(Counter(i['domain'] for i in items))}")


def show_db():
    print("\n--- database ---")
    db = path("db")
    if not db.exists():
        print(f"{db} not created yet (first run will create it)")
        return
    conn = sqlite3.connect(db)
    for table in ("responses", "judgments"):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {n} rows")
        except sqlite3.OperationalError:
            print(f"{table}: not created yet")
    try:
        rows = conn.execute("SELECT phase, model, COUNT(*) FROM responses "
                            "GROUP BY phase, model").fetchall()
        for phase, model, n in rows:
            print(f"  {phase:<12} {model:<15} {n}")
    except sqlite3.OperationalError:
        pass
    conn.close()


def ping(ready):
    print("\n--- ping (real calls) ---")
    for key in ready:
        text = models.ask(key, "Reply with the single word: ready",
                          max_tokens=5)
        status = "OK  " if text else "FAIL"
        print(f"{status}{key:<15} {str(text).strip()[:40]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true",
                    help="send one tiny real request per ready model")
    args = ap.parse_args()
    ready = show_models()
    show_data()
    show_db()
    if args.ping and ready:
        ping(ready)


if __name__ == "__main__":
    main()
