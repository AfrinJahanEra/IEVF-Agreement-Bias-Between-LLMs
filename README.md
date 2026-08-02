# IEVF — Agreement Bias Between LLMs

Does one LLM change another LLM's mind under peer pressure — and can the
**IEVF** framework (with the **EGDA** auditor) stop it?

**Benchmarks:** MoReBench (moral dilemmas + expert checklists), SimpleBench
(10 verifiable probe questions) and **AschBench** (our own 100 pressure-ready
items). **No training, no LoRA** — pure inference-time study via APIs.

**Swapping LLMs needs no code change:** every model is a block in
`config.yaml` plus a token in `.env`. A model without a token is skipped
automatically, and each model is graded by an automatically chosen judge of a
different family.

## Project map

```
research_ievf/
├── run_pipeline.py        ← runs every phase (fetch-data/baseline/exp1/exp2/analyze)
├── check_setup.py         ← preflight: tokens, judges, data, database (run this first)
├── fetch_morebench.py     ← one-time: downloads MoReBench into data/
├── make_reports.py        ← regenerates the two PDF reports from this README's content
├── config.yaml            ← models + judges + settings (edit this to swap LLMs)
├── .env                   ← your API keys (create from .env.example; gitignored, never commit it)
├── src/
│   ├── config.py          loads config + keys
│   ├── models.py          ask(key, prompt) — one door to every model/judge (+ --dry-run fakes)
│   ├── storage.py         SQLite: every response + every grading (resume-safe)
│   ├── data_loader.py     MoReBench / SimpleBench / AschBench → one format
│   ├── prompts.py         all prompt templates
│   ├── judge.py           YES/NO grading per checklist point + 0-100 score
│   ├── peers.py           builds peer messages (P1-P6, scripted groups)
│   ├── ievf.py            the IEVF+EGDA gate
│   ├── pipeline.py        ONE engine; each experiment is a condition generator
│   └── analyze.py         metrics tables + figures
├── benchmark/                    ← AschBench sources (benchmark/items/*.jsonl) + build_release.py
├── data/
│   ├── asch_bench_public.json    ← our merged benchmark release
│   ├── morebench_public.csv      ← from fetch_morebench.py (public, no gating)
│   ├── morebench_theory.csv      ← ditto, theory subset
│   ├── simple_bench_public.json  ← already in the repo
│   └── processed/items.jsonl     ← the question bank (built by --phase fetch-data)
├── logs/results.db               ← ALL real-run raw data lives here
│   (a --dry-run always writes to logs/dryrun_results.db instead — never the real file)
└── reports/                      ← final tables + plots + the two PDFs
    (a --dry-run analyze writes to reports/dryrun/ instead — never the real files)
```

## Step-by-step: what to do

### 0. Install (once)
```powershell
cd research_ievf
pip install -r requirements.txt
copy .env.example .env     # then open .env and paste your keys
```
`.env` is listed in `.gitignore` — it never gets committed. Still, treat any
key you paste anywhere (chat, screenshare, etc.) as compromised and rotate it
afterward; don't rely on gitignore alone for that.

Key names by provider (only fill in what you have — anything missing is
just skipped):

| Provider | `.env` variable |
|---|---|
| OpenAI (gpt-5) | `OPENAI_API_KEY` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| Google (Gemini) | `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Qwen (Alibaba DashScope) | `DASHSCOPE_API_KEY` |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Judge (`judge-deepseek` in `config.yaml`) | `JUDGE_API_KEY` — token only, the model/endpoint are set in `config.yaml` |
| Hugging Face (optional, raises MoReBench rate limit) | `HF_TOKEN` |

### 1. Preflight check (free, no API calls)
```powershell
python check_setup.py          # which models are ready, which judge each gets
python check_setup.py --ping   # + one tiny real call per model (a few tokens)
```
Only models whose token is in `.env` take part in the study. Each ready
model needs at least one *other* ready model from a different `family:` in
`config.yaml` to act as its judge — `check_setup.py` prints the pairing it
picked, or a warning if none is available.

### 2. Get the real data
```powershell
python fetch_morebench.py                     # once; public dataset, HF_TOKEN optional
python run_pipeline.py --phase fetch-data --simplebench --aschbench
```
MoReBench (250 sampled scenarios + 150-item theory subset) + 10 SimpleBench
questions + our 100 AschBench items → 510 total.
→ Check `data/processed/items.jsonl` — one question per line.

### 3. Free smoke test (fake answers, zero cost)
```powershell
python run_pipeline.py --phase baseline --dry-run
python run_pipeline.py --phase exp1 --dry-run
python run_pipeline.py --phase exp2 --dry-run
python run_pipeline.py --phase exp1 --dry-run --mit
python run_pipeline.py --phase exp2 --dry-run --mit
python run_pipeline.py --phase analyze --dry-run
```
Exercises the entire pipeline — prompts, storage, grading, the IEVF gate —
with placeholder text instead of real API calls. Writes only to
`logs/dryrun_results.db` and `reports/dryrun/`, so it can never mix with or
overwrite real study data. Always run this before spending money on a new
model/config combination.

### 4. Small REAL test (one cheap model, 5 items — costs a few cents)
```powershell
python run_pipeline.py --phase baseline --models qwen --limit 5
```
Open `logs/results.db` with **DB Browser for SQLite** (free app) →
table `responses` = model answers, table `judgments` = YES/NO per
checklist point. Read a few by eye: does the judge grade sensibly?

### 5. Full study (run in this order; each is resume-safe)
```powershell
python run_pipeline.py --phase baseline      # Step 1: private answers
python run_pipeline.py --phase exp1          # Step 2: one-peer test P1-P6
python run_pipeline.py --phase exp2          # Step 3: group pressure
python run_pipeline.py --phase exp1 --mit    # Step 6: re-run with IEVF
python run_pipeline.py --phase exp2 --mit    # Step 7: re-run with IEVF
python run_pipeline.py --phase analyze       # Step 8: tables + figures
python make_reports.py                       # regenerate the two PDF reports
```
Crash? Power cut? Just run the same command again — finished work is skipped.
Use `--models x,y` to run a subset at a time (spreads cost/rate-limits across
models) and `--limit N` to cap items per invocation.

### 6. Where to look at results
| What | Where |
|---|---|
| Every prompt + full answer | `logs/results.db` → table `responses` (DB Browser) |
| Every grading | `logs/results.db` → table `judgments` |
| Drop/gain/bad rates per condition | `reports/metrics_by_condition.csv` |
| **The paper's Table 1** (before vs after IEVF) | `reports/before_after.csv` |
| What the IEVF gate allowed/blocked, and why | `reports/gate_decisions.csv` |
| Stuck rate per group size (hysteresis) | `reports/hysteresis.csv` |
| Plots (dose-response, before/after, per-model) | `reports/figures/*.png` |
| Readable reports | `reports/report_1_pipeline.pdf`, `report_2_experiment_flow.pdf` |

### 7. Human check of the judge (needed for the paper)
Open `judgments` in DB Browser, export 10% to CSV, mark YES/NO yourself,
compute agreement (Cohen's κ ≥ 0.7 target).

### 8. Report writing
- **Methods** = this README + `config.yaml` + `src/prompts.py` (cite as protocol)
- **Results Table 1** = `reports/before_after.csv`
- **Results Figure 1** = `reports/figures/dose_response.png`
- **Results Figure 2** = `reports/figures/before_after.png`
- **Robustness** = per-model rows of `metrics_by_condition.csv`

## Troubleshooting real API errors

Free-tier limits and billing issues are the most common blockers, and they
come from the provider, not this pipeline — `run_pipeline.py` retries 3
times then logs the failure and moves on (never crashes the run):

| Error (in the `[warn]`/`[error]` log lines) | Meaning | Fix |
|---|---|---|
| `429 RESOURCE_EXHAUSTED` (Google) | Free-tier daily/per-minute quota hit — some free-tier Gemini models cap as low as 20 requests/day | Enable billing on that Google Cloud project, wait for the daily reset, or use a different model/key |
| `402 Insufficient Balance` (DeepSeek) | Account has no funds | Top up at platform.deepseek.com |
| `Your credit balance is too low` (Anthropic) | Account has no funds | Top up at console.anthropic.com |
| `401 Incorrect API key` | Key is wrong, truncated, or revoked | Re-copy the key from the provider's console |
| `No independent judge available for X` | Every configured judge is missing a token, or shares X's `family:` | Add a token for a judge of a different family in `.env` |

`python check_setup.py --ping` catches all of these cheaply (a few tokens
each) before you commit to a real run.

## Rules that protect the science
1. Judge model is **never the same family** as the model being graded —
   enforced in code by `models.judge_for()` using `family:` in `config.yaml`.
2. Temperature stays 0 (`config.yaml`) so results are repeatable.
3. Never delete `logs/results.db` between before/after runs — the
   comparison needs both.
4. Peer messages are always real model answers; if no peer answer exists yet,
   the cell is skipped rather than filled with invented text.
5. `--dry-run` never touches real data — it's isolated to
   `logs/dryrun_results.db` / `reports/dryrun/` by design.
