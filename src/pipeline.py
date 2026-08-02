"""The experiment phases. Every function follows the same loop:
read item -> build prompt -> ask model -> save -> grade -> next item.

Resume-safe everywhere: finished (phase, model, item, condition, round)
combinations are skipped automatically.
"""
from tqdm import tqdm

from . import ievf, peers, storage
from .config import CFG, env
from .judge import grade_response
from .models import ask
from .prompts import (BASELINE, BASELINE_SIMPLEBENCH, EXPOSURE,
                      GROUP_ROUND, HYSTERESIS_PROBE, NEUTRAL_REASK)

JUDGE = env("JUDGE_MODEL", "deepseek-chat")


def _item_prompt(item):
    if item["domain"] == "simplebench":
        return BASELINE_SIMPLEBENCH.format(scenario=item["prompt"])
    return BASELINE.format(scenario=item["prompt"])


def _run_and_grade(phase, model, item, condition, round_, prompt, mock):
    """Ask + save + grade one response. Returns the response id."""
    if storage.already_done(phase, model, item["item_id"], condition, round_):
        rec = storage.get_response(phase, model, item["item_id"], condition,
                                   round_)
        return rec["id"]
    text = ask(model, prompt, mock=mock)
    if text is None:
        return None
    rid = storage.save_response(phase, model, item["item_id"], condition,
                                round_, prompt, text)
    grade_response(rid, item["prompt"], text, item["criteria"], JUDGE,
                   mock=mock)
    return rid


# ----------------------------------------------------------------------
# STEP 1 - Baseline: every model answers every item alone
# ----------------------------------------------------------------------
def run_baseline(items, models, mock=False, limit=None):
    items = items[:limit] if limit else items
    for model in models:
        for item in tqdm(items, desc=f"baseline/{model}"):
            _run_and_grade("baseline", model, item, "private", 0,
                           _item_prompt(item), mock)


# ----------------------------------------------------------------------
# STEP 2 - Experiment 1: one peer tries to change one model (cells P1-P6)
# ----------------------------------------------------------------------
def run_exp1(items, models, mock=False, limit=None, mit=False):
    phase = "exp1_mit" if mit else "exp1"
    items = items[:limit] if limit else items
    cells = CFG["experiment1"]["cells"]
    for model in models:
        for item in tqdm(items, desc=f"{phase}/{model}"):
            private = storage.get_response("baseline", model,
                                           item["item_id"], "private", 0)
            if not private:
                continue  # baseline missing for this pair - skip
            for cell in cells:
                peer_block = peers.build_peer_block(cell, model, item, models)
                if peer_block is None:
                    prompt = NEUTRAL_REASK.format(scenario=item["prompt"])
                else:
                    prompt = EXPOSURE.format(scenario=item["prompt"],
                                             peer_block=peer_block)
                rid = _run_and_grade(phase, model, item, cell, 0, prompt,
                                     mock)
                if rid is None or not mit:
                    continue
                # IEVF gate: reject unjustified changes -> save final answer
                new = storage.get_response(phase, model, item["item_id"],
                                           cell, 0)
                allow, verdict = ievf.gate(
                    item, private["id"], new["id"], private["response"],
                    peer_block, new["response"], JUDGE, mock=mock)
                final_text = new["response"] if allow else private["response"]
                storage.save_response(phase, model, item["item_id"],
                                      cell + "_final", 0,
                                      f"IEVF gate verdict: {verdict}",
                                      final_text)


# ----------------------------------------------------------------------
# STEP 3 - Experiment 2: group pressure (scripted majority) + probes
# ----------------------------------------------------------------------
def run_exp2(items, models, mock=False, limit=None, mit=False):
    phase = "exp2_mit" if mit else "exp2"
    items = items[:limit] if limit else items
    ks = CFG["experiment2"]["majority_sizes"]
    rounds = CFG["experiment2"]["rounds"]
    for model in models:
        for item in tqdm(items, desc=f"{phase}/{model}"):
            for k in ks:
                cond = f"G_k{k}"
                group_block = peers.scripted_group(k, item)
                for r in range(1, rounds + 1):
                    prompt = GROUP_ROUND.format(scenario=item["prompt"],
                                                group_block=group_block,
                                                round_no=r)
                    _run_and_grade(phase, model, item, cond, r, prompt, mock)
                # Hysteresis probe: fresh chat, no peer content
                _run_and_grade(phase, model, item, cond + "_probe", 0,
                               HYSTERESIS_PROBE.format(
                                   scenario=item["prompt"]), mock)


def run(items, models, phase, mock=False, limit=None, mit=False):
    storage.init_db()
    if phase == "baseline":
        run_baseline(items, models, mock, limit)
    elif phase == "exp1":
        run_exp1(items, models, mock, limit, mit)
    elif phase == "exp2":
        run_exp2(items, models, mock, limit, mit)
    else:
        raise ValueError(f"unknown phase {phase}")
