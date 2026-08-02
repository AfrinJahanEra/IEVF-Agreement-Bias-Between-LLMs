"""Reads the results database and produces:
  reports/metrics_by_condition.csv  - drop/gain/bad-point rates per condition
  reports/before_after.csv          - IEVF on vs off (the paper's Table 1)
  reports/figures/*.png             - the three headline plots
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import storage
from .config import path
from .judge import scenario_score


def _profiles(df_j, judge_model):
    """{(response_id): (mets, weights, dimensions)} for one judge."""
    out = {}
    df = df_j[df_j.judge_model == judge_model]
    for rid, g in df.groupby("response_id"):
        g = g.sort_values("criterion_idx")
        out[rid] = (g.met.tolist(), g.weight.tolist(),
                    g.dimension.tolist())
    return out


def _diff_rates(priv, new):
    """Compare private vs new profile -> drop/gain/bad rates + score drift."""
    pm, pw, pdim = priv
    nm, nw, _ = new
    priv_pos = [i for i, (m, w) in enumerate(zip(pm, pw)) if w > 0 and m == 1]
    unmet_pos = [i for i, (m, w) in enumerate(zip(pm, pw)) if w > 0 and m == 0]
    neg_idx = [i for i, w in enumerate(pw) if w < 0]

    dropped = sum(1 for i in priv_pos if i < len(nm) and nm[i] == 0)
    gained = sum(1 for i in unmet_pos if i < len(nm) and nm[i] == 1)
    bad = sum(1 for i in neg_idx if i < len(nm) and nm[i] == 1
              and pm[i] == 0)

    drop_rate = dropped / len(priv_pos) if priv_pos else 0.0
    gain_rate = gained / len(unmet_pos) if unmet_pos else 0.0
    bad_rate = bad / len(neg_idx) if neg_idx else 0.0
    drift = scenario_score(nm, nw) - scenario_score(pm, pw)
    return drop_rate, gain_rate, bad_rate, drift


def build_metrics(judge_model, phases=("exp1", "exp2")):
    resp = storage.to_dataframe("responses")
    judg = storage.to_dataframe("judgments")
    prof = _profiles(judg, judge_model)
    rows = []
    for phase in phases:
        df = resp[resp.phase == phase]
        base = resp[resp.phase == "baseline"]
        for (model, item, cond), g in df.groupby(["model", "item_id",
                                                  "condition"]):
            p = base[(base.model == model) & (base.item_id == item)]
            if p.empty:
                continue
            pid, nid = p.iloc[0].id, g.iloc[0].id
            if pid not in prof or nid not in prof:
                continue
            dr, gr, br, drift = _diff_rates(prof[pid], prof[nid])
            rows.append({"phase": phase, "model": model, "item_id": item,
                         "condition": cond, "drop_rate": dr,
                         "gain_rate": gr, "bad_rate": br,
                         "score_drift": drift})
    return pd.DataFrame(rows)


def run_analysis(judge_model):
    out_dir = path("reports")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plain = build_metrics(judge_model, phases=("exp1", "exp2"))
    mit = build_metrics(judge_model, phases=("exp1_mit", "exp2_mit"))

    by_cond = (plain.groupby(["phase", "condition"])[
        ["drop_rate", "gain_rate", "bad_rate", "score_drift"]]
        .mean().reset_index())
    by_cond.to_csv(out_dir / "metrics_by_condition.csv", index=False)

    if not mit.empty:
        m1 = plain.groupby("phase")[["drop_rate", "gain_rate", "bad_rate",
                                     "score_drift"]].mean()
        m2 = mit.groupby("phase")[["drop_rate", "gain_rate", "bad_rate",
                                   "score_drift"]].mean()
        comp = pd.concat({"before_IEVF": m1, "after_IEVF": m2}, axis=1)
        comp.to_csv(out_dir / "before_after.csv")

    # Figure 1: dose-response (majority size -> drop rate), exp2 only
    exp2 = plain[plain.phase == "exp2"].copy()
    if not exp2.empty:
        exp2["k"] = exp2.condition.str.extract(r"G_k(\d)").astype(float)
        curve = exp2.dropna(subset=["k"]).groupby("k").drop_rate.mean()
        curve.plot(marker="o")
        plt.xlabel("Number of opposing peers (k)")
        plt.ylabel("Consideration Drop Rate")
        plt.title("Dose-response: more peers -> more dropping")
        plt.savefig(fig_dir / "dose_response.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    # Figure 2: before/after bars (only if mitigation ran)
    if not mit.empty:
        ax = (pd.DataFrame({
            "before": plain.drop_rate.mean(),
            "after": mit.drop_rate.mean()}, index=["drop_rate"]).T)
        ax.plot(kind="bar", rot=0)
        plt.ylabel("Consideration Drop Rate")
        plt.title("IEVF before vs after")
        plt.savefig(fig_dir / "before_after.png", dpi=150,
                    bbox_inches="tight")
        plt.close()

    print(f"Reports written to {out_dir}")
