"""One function to talk to every model: ask(model_key, prompt).

All providers hide behind the same interface. Retries 3 times on failure,
then returns None (the pipeline logs a gap and continues — one bad call
must never kill a 10,000-call run).

DRY-RUN mode: ask(..., mock=True) returns a fake deterministic answer, so the
whole pipeline can be tested with zero API keys and zero cost.
"""
import time

from .config import CFG, env

_clients = {}


def _openai_compatible(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]
    if provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=env("OPENAI_API_KEY"))
    elif provider == "anthropic":
        import anthropic
        c = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"))
    elif provider == "google":
        from google import genai
        c = genai.Client(api_key=env("GOOGLE_API_KEY"))
    elif provider == "deepseek":
        c = _openai_compatible("https://api.deepseek.com", env("DEEPSEEK_API_KEY"))
    elif provider == "qwen":
        c = _openai_compatible(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            env("DASHSCOPE_API_KEY"))
    elif provider == "moonshot":
        c = _openai_compatible("https://api.moonshot.ai/v1", env("MOONSHOT_API_KEY"))
    elif provider == "judge":
        c = _openai_compatible(env("JUDGE_BASE_URL"), env("JUDGE_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    _clients[provider] = c
    return c


def _call(provider: str, model: str, prompt: str, temperature: float,
          max_tokens: int) -> str:
    client = _get_client(provider)
    if provider == "anthropic":
        r = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if provider == "google":
        r = client.models.generate_content(model=model, contents=prompt)
        return r.text
    # openai / deepseek / qwen / moonshot / judge — all OpenAI-compatible
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def _mock_response(prompt: str) -> str:
    """Deterministic fake answer for dry-runs (mentions common moral points)."""
    return (
        "Let me think through this dilemma carefully. "
        "I should weigh honesty and fairness, consider the harm to others, "
        "and respect the person's trust. Balancing these, my recommendation "
        "is to act honestly while minimizing harm.\n"
        "Final answer: I would act honestly and minimize harm."
    )


def ask(model_key: str, prompt: str, mock: bool = False,
        temperature: float = None, max_tokens: int = None,
        _retries: int = 3):
    """Send one prompt to one model. Returns text or None on final failure."""
    if mock:
        return _mock_response(prompt)
    mc = CFG["models"][model_key]
    temperature = CFG["generation"]["temperature"] if temperature is None else temperature
    max_tokens = CFG["generation"]["max_tokens"] if max_tokens is None else max_tokens
    for attempt in range(_retries):
        try:
            return _call(mc["provider"], mc["model"], prompt, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 - log and retry, never crash a run
            wait = 2 ** attempt * 2
            print(f"  [warn] {model_key} attempt {attempt + 1} failed: {e}. "
                  f"Retry in {wait}s")
            time.sleep(wait)
    print(f"  [error] {model_key} gave up after {_retries} attempts")
    return None


def ask_judge(prompt: str, mock: bool = False, _retries: int = 3):
    """Send one prompt to the judge model (grading + EGDA)."""
    if mock:
        return "YES"  # dry-run: judge says everything is fulfilled
    for attempt in range(_retries):
        try:
            return _call("judge", env("JUDGE_MODEL", "deepseek-chat"),
                         prompt, 0.0, 1024)
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt * 2
            print(f"  [warn] judge attempt {attempt + 1} failed: {e}")
            time.sleep(wait)
    return None
