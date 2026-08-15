"""One function to talk to every model: ask(key, prompt).

Everything is driven by config.yaml + .env - no provider logic is hard-coded
per company. A model or judge is just a spec:

    {provider, model, api_key_env, base_url (optional), family}

So swapping an LLM = editing config.yaml and putting its token in .env.

One exception: provider=huggingface runs the model LOCALLY via transformers
(weights download once from the Hub, then pure local inference) - it needs
no api_key_env at all.

Retries 3 times on failure, then returns None (the pipeline logs the gap and
continues - one bad call must never kill a 10,000-call run).
"""
import json
import time
from collections import OrderedDict

from .config import CFG, env

_clients = {}

# Local HF models are big: keep only the last MAX_LOCAL_MODELS resident
# (a study model + its judge is exactly 2). Older ones are evicted and
# their VRAM freed - this lets a 15 GB Colab GPU cycle through many models.
# Raise it in config.yaml (local.max_resident) when fewer/smaller models
# all fit at once - then the reload bars disappear entirely.
_local_models = OrderedDict()
MAX_LOCAL_MODELS = int(CFG.get("local", {}).get("max_resident", 2))

DRY_RUN = False  # set by run_pipeline.py --dry-run: fake answers, no API calls


def _fake_answer(key: str, prompt: str) -> str:
    """Deterministic stand-in used under DRY_RUN - exercises every prompt/
    parse/storage code path without a network call or any cost."""
    if key in CFG.get("judges", {}):
        if "allow_flip" in prompt:  # the EGDA auditor call expects JSON
            return json.dumps({"new_evidence": False, "driver": "unclear",
                               "allow_flip": False})
        return "YES"  # per-criterion rubric grading expects YES/NO
    return ("[dry-run placeholder response]\n"
            "Final recommendation: dry-run placeholder\n"
            "Final answer: A")


def _registry():
    """All callable keys: study models + judges."""
    reg = dict(CFG["models"])
    reg.update(CFG.get("judges", {}))
    return reg


def spec(key: str) -> dict:
    reg = _registry()
    if key not in reg:
        raise KeyError(f"Unknown model key {key!r}. Known: {list(reg)}")
    return reg[key]


def family(key: str) -> str:
    return spec(key).get("family", key)


def token(key: str) -> str:
    return env(spec(key).get("api_key_env", ""))


def is_local(key: str) -> bool:
    """Local Hugging Face models need no token at all."""
    return spec(key)["provider"] == "huggingface"


def has_token(key: str) -> bool:
    return is_local(key) or bool(token(key))


def study_models(only=None):
    """Model keys to run: those with a token present (optionally filtered).
    This is what makes 'just change the token' work - a model with no key in
    .env is simply not part of the run."""
    keys = list(only) if only else list(CFG["models"])
    unknown = [k for k in keys if k not in CFG["models"]]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}. "
                         f"Known: {list(CFG['models'])}")
    ready = [k for k in keys if has_token(k)]
    skipped = [k for k in keys if not has_token(k)]
    for k in skipped:
        print(f"  [skip] {k}: no {spec(k)['api_key_env']} in .env")
    if not ready:
        raise SystemExit(
            "No model has a token in .env. Add at least one key "
            "(see .env.example) and re-run.")
    return ready


def judge_for(model_key: str) -> str:
    """First configured judge with a token and a DIFFERENT family than the
    target - the independence rule, applied automatically."""
    target = family(model_key)
    for jk in CFG.get("judges", {}):
        if has_token(jk) and family(jk) != target:
            return jk
    raise SystemExit(
        f"No independent judge available for {model_key} (family {target}). "
        "Add a judge of another family in config.yaml + its token in .env.")


# ----------------------------------------------------------------------
# transport: three API styles cover every provider
# ----------------------------------------------------------------------
def _client(sp: dict):
    if sp["provider"] == "huggingface":
        return _load_local(sp)  # has its own LRU cache + VRAM eviction
    cache_key = (sp["provider"], sp.get("base_url"), sp.get("api_key_env"))
    if cache_key in _clients:
        return _clients[cache_key]
    api_key = env(sp.get("api_key_env", ""))
    provider = sp["provider"]
    if provider == "openai":
        from openai import OpenAI
        c = (OpenAI(api_key=api_key, base_url=sp["base_url"])
             if sp.get("base_url") else OpenAI(api_key=api_key))
    elif provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=api_key)
    elif provider == "google":
        from google import genai
        c = genai.Client(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. Use openai | anthropic | google"
            " | huggingface (local).")
    _clients[cache_key] = c
    return c


def _load_local(sp: dict):
    """Load a Hugging Face model once, run it fully locally - no API.
    Uses the GPU when one is available, otherwise float32 on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = sp["model"]
    cache_key = sp["model"]
    if cache_key in _local_models:
        _local_models.move_to_end(cache_key)
        return _local_models[cache_key]
    # evict least-recently-used local models before loading a new one
    while len(_local_models) >= MAX_LOCAL_MODELS:
        old_key, _ = _local_models.popitem(last=False)
        print(f"  [hf] evicting {old_key} to make room")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [hf] loading {name} into memory (weights come from the "
          f"local HF cache - downloaded at most once per runtime session)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:  # transformers >= 5.0 calls it dtype; older versions torch_dtype
        net = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        net = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    # from_pretrained loads to CPU by default - move it to the GPU ourselves
    net.to(device)
    net.eval()
    bundle = {"tokenizer": tok, "model": net, "device": device}
    _local_models[cache_key] = bundle
    return bundle


def _call(sp: dict, prompt: str, temperature: float, max_tokens: int) -> str:
    if sp["provider"] == "huggingface":
        return _call_local(sp, prompt, temperature, max_tokens)
    client = _client(sp)
    if sp["provider"] == "anthropic":
        r = client.messages.create(
            model=sp["model"], max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if sp["provider"] == "google":
        r = client.models.generate_content(model=sp["model"], contents=prompt)
        return r.text
    r = client.chat.completions.create(
        model=sp["model"], temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def _call_local(sp: dict, prompt: str, temperature: float,
                max_tokens: int) -> str:
    """One local generation call. temperature=0 -> greedy (repeatable,
    matching the API path)."""
    import torch
    b = _client(sp)  # cached local bundle
    _local_models.move_to_end(sp["model"])
    tok, net, device = b["tokenizer"], b["model"], b["device"]
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
        return_dict=True).to(device)
    # prompts can be long (peer blocks); the small models we use fit far
    # more than this - truncate defensively rather than crash
    if inputs["input_ids"].shape[1] > 4096:
        inputs = {k: v[:, :4096] for k, v in inputs.items()}
    gen = dict(max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
               # small models fall into repetition loops that burn through
               # the whole token cap and stall the run - penalize repeats
               repetition_penalty=1.1)
    if temperature and temperature > 0:
        gen.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        out = net.generate(**inputs, **gen)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def ask(key: str, prompt: str, temperature=None, max_tokens=None,
        _retries: int = 3):
    """Send one prompt to one model or judge. Text on success, None if all
    retries fail."""
    if DRY_RUN:
        return _fake_answer(key, prompt)
    sp = spec(key)
    gen = CFG["judging"] if key in CFG.get("judges", {}) else CFG["generation"]
    temperature = gen["temperature"] if temperature is None else temperature
    max_tokens = gen["max_tokens"] if max_tokens is None else max_tokens
    for attempt in range(_retries):
        try:
            return _call(sp, prompt, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 - log and retry, never crash
            wait = 2 ** attempt * 2
            print(f"  [warn] {key} attempt {attempt + 1} failed: {e}. "
                  f"Retry in {wait}s")
            time.sleep(wait)
    print(f"  [error] {key} gave up after {_retries} attempts")
    return None
