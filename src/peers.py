"""Builds the peer messages shown to target models.

Peer content is ALWAYS real: it comes from another model's actual baseline
answer (P1, P2, P4, P6). If no other model has answered an item yet, the
cell returns UNAVAILABLE and the driver skips it - no invented peer text
ever enters the study. Experiment 2 uses the item's own scripted majority
(AschBench) or one standard template.
"""
import random

from . import storage
from .prompts import PEER_BLOCKS, SCRIPTED_PEER

_rng = random.Random(42)

# Explicit signals returned by build_peer_block()
NEUTRAL = "__neutral__"          # P5 control: show no peer content at all
UNAVAILABLE = "__unavailable__"  # no real peer answer exists yet -> skip


def build_peer_block(cell, target_key, item, model_keys):
    """Return the peer text for one Experiment-1 cell, or the NEUTRAL /
    UNAVAILABLE signal."""
    if cell == "P5":
        return NEUTRAL

    private = storage.get_response("baseline", target_key, item["item_id"],
                                   "private", 0)
    if cell == "P6":  # phantom-self: its own answer shown as a peer
        if not (private and private["response"]):
            return UNAVAILABLE
        peer_text = private["response"]
    else:
        other = _other_model_answer("baseline", target_key, item["item_id"],
                                    model_keys)
        if not other:
            return UNAVAILABLE
        if cell == "P2":  # narrowed peer: conclusion only, no reasoning
            peer_text = (other.split("Final recommendation:")[-1].strip()
                         if "Final recommendation:" in other else other[:300])
        else:  # P1, P3, P4: another model's full answer
            peer_text = other
    return PEER_BLOCKS[cell].format(peer_text=peer_text)


def _other_model_answer(phase, target_key, item_id, model_keys):
    """A baseline answer from a different model for this item, or None."""
    candidates = [m for m in model_keys if m != target_key]
    _rng.shuffle(candidates)
    for m in candidates:
        rec = storage.get_response(phase, m, item_id, "private", 0)
        if rec and rec["response"]:
            return rec["response"]
    return None


def scripted_group(k, item=None, stance="take the easier option",
                   reason=None):
    """k pressure messages for Experiment 2.

    AschBench items carry their own majority script - use it (item-specific
    pressure). Everything else falls back to the generic template.
    """
    if item and item.get("pressure", {}).get("majority", {}).get("script"):
        return "\n".join(item["pressure"]["majority"]["script"][:k])
    reason = reason or ("it benefits the most people involved and the risks "
                        "are small")
    return "\n".join(SCRIPTED_PEER.format(i=i + 1, stance=stance,
                                          reason=reason)
                     for i in range(k))
