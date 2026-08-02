"""The experiment engine.

ONE loop runs every experiment:
    read item -> build prompt -> ask model -> save -> grade -> (IEVF gate)

An "experiment" is only a generator of (condition, round, prompt, peer_text)
tuples, so Experiment 1 and Experiment 2 share identical machinery: the same
asking, saving, grading and gating code path. Adding an experiment means
adding one generator to _EXPERIMENTS - nothing else.

Switching LLMs changes nothing here: models come from config.yaml and their
tokens from .env, and each target model automatically gets an independent
judge (different family).

Resume-safe everywhere: finished (phase, model, item, condition, round)
combinations are skipped automatically.
"""
import json

from tqdm import tqdm

from . import ievf, models, peers, storage
from .config import CFG
from .judge import grade_response
from .prompts import (BASELINE, BASELINE_SIMPLEBENCH, EXPOSURE, GROUP_ROUND,
                      HYSTERESIS_PROBE, NEUTRAL_REASK)


def _item_prompt(item):
    if item["domain"] == "simplebench":
        return BASELINE_SIMPLEBENCH.format(scenario=item["prompt"])
    return BASELINE.format(scenario=item["prompt"])


def _run_and_grade(phase, model_key, item, condition, round_, prompt,
                   judge_key):
    """Ask + save + grade one response. Returns the response id or None."""
    if storage.already_done(phase, model_key, item["item_id"], condition,
                            round_):
        rec = storage.get_response(phase, model_key, item["item_id"],
                                   condition, round_)
        return rec["id"]
    text = models.ask(model_key, prompt)
    if text is None:
        return None
    rid = storage.save_response(phase, model_key, item["item_id"], condition,
                                round_, prompt, text)
    grade_response(rid, item["prompt"], text, item["criteria"], judge_key)
    return rid


# ----------------------------------------------------------------------
# Experiment definitions: each yields (condition, round, prompt, peer_text)
# ----------------------------------------------------------------------
def _baseline_conditions(item, model_key, model_keys):
    """Every model answers every item alone - its private answer."""
    yield "private", 0, _item_prompt(item), None


def _exp1_conditions(item, model_key, model_keys):
    """One peer answer, six framings (P1-P6)."""
    for cell in CFG["experiment1"]["cells"]:
        peer_block = peers.build_peer_block(cell, model_key, item, model_keys)
        if peer_block is peers.UNAVAILABLE:
            continue  # no other model has answered this item yet
        if peer_block is peers.NEUTRAL:
            yield cell, 0, NEUTRAL_REASK.format(scenario=item["prompt"]), None
        else:
            yield cell, 0, EXPOSURE.format(scenario=item["prompt"],
                                           peer_block=peer_block), peer_block


def _exp2_conditions(item, model_key, model_keys):
    """A scripted majority of k members, over several rounds, then a
    pressure-free probe (hysteresis)."""
    for k in CFG["experiment2"]["majority_sizes"]:
        group_block = peers.scripted_group(k, item)
        cond = f"G_k{k}"
        for r in range(1, CFG["experiment2"]["rounds"] + 1):
            yield cond, r, GROUP_ROUND.format(
                scenario=item["prompt"], group_block=group_block,
                round_no=r), group_block
        yield (cond + "_probe", 0,
               HYSTERESIS_PROBE.format(scenario=item["prompt"]), None)


_EXPERIMENTS = {
    "baseline": _baseline_conditions,
    "exp1": _exp1_conditions,
    "exp2": _exp2_conditions,
}


# ----------------------------------------------------------------------
# The single driver used by every experiment
# ----------------------------------------------------------------------
def run(items, model_keys, phase, limit=None, mit=False):
    """Run one experiment for every model over every item.

    phase: 'baseline' | 'exp1' | 'exp2'
    mit:   True -> IEVF+EGDA gate is applied and the phase is stored
           under '<phase>_mit' so before/after stay comparable.
    """
    if phase not in _EXPERIMENTS:
        raise ValueError(f"unknown phase {phase}. "
                         f"Known: {list(_EXPERIMENTS)}")
    if phase == "baseline" and mit:
        raise ValueError("baseline has no peer pressure, so --mit is "
                         "meaningless here")
    storage.init_db()
    conditions = _EXPERIMENTS[phase]
    stored_phase = f"{phase}_mit" if mit else phase
    items = items[:limit] if limit else items

    for model_key in model_keys:
        judge_key = models.judge_for(model_key)
        print(f"{stored_phase}: {model_key} graded by {judge_key}")
        for item in tqdm(items, desc=f"{stored_phase}/{model_key}"):
            private = None
            if phase != "baseline":
                private = storage.get_response("baseline", model_key,
                                               item["item_id"], "private", 0)
                if not private:
                    continue  # baseline missing for this pair - skip
            for cond, round_, prompt, peer_text in conditions(
                    item, model_key, model_keys):
                rid = _run_and_grade(stored_phase, model_key, item, cond,
                                     round_, prompt, judge_key)
                if rid is None or not mit or peer_text is None:
                    continue
                # IEVF gate: keep the change only if it is evidence-driven
                new = storage.get_response(stored_phase, model_key,
                                           item["item_id"], cond, round_)
                allow, verdict = ievf.gate(
                    item, private["id"], new["id"], private["response"],
                    peer_text, new["response"], judge_key)
                final_text = new["response"] if allow else private["response"]
                storage.save_response(stored_phase, model_key,
                                      item["item_id"], f"{cond}_final",
                                      round_,
                                      "IEVF gate verdict: "
                                      + json.dumps(verdict),
                                      final_text)
