"""Quick health check: python verify_setup.py
Prints database counts and confirms AschBench pressure scripts are used."""
import sqlite3

conn = sqlite3.connect("logs/results.db")

print("--- database ---")
for table in ("responses", "judgments"):
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows")
print("phases:", [r[0] for r in
                   conn.execute("SELECT DISTINCT phase FROM responses")])

print("\n--- AschBench wiring ---")
row = conn.execute(
    "SELECT prompt FROM responses WHERE phase='exp2' AND model='qwen'"
    " AND item_id='asch_0003' AND condition='G_k3' AND round=1").fetchone()
if row and "Member 2: B, clearly" in row[0]:
    print("OK: exp2 used the item's own majority script")
else:
    print("MISSING: run baseline+exp2 dry-run on the sample+aschbench items")
