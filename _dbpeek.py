import sqlite3

c = sqlite3.connect("logs/results.db")
q = lambda s: c.execute(s).fetchall()
print("rows by phase:")
for r in q("select phase, count(*) from responses group by phase order by phase"):
    print("  ", r)
print("distinct items in DB:", q("select count(distinct item_id) from responses")[0][0])
print("models:", q("select distinct model from responses"))
print("items per phase (exp):")
for r in q("select phase, count(distinct item_id) from responses "
           "where phase in ('exp1','exp2') group by phase"):
    print("  ", r)
