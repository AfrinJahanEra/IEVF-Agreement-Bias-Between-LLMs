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
├── run_pipeline.py        ← the only file you run
├── check_setup.py         ← preflight: tokens, judges, data, database
├── config.yaml            ← models + judges + settings (edit this to swap LLMs)
├── .env                   ← your API keys (create from .env.example)
├── src/
│   ├── config.py          loads config + keys
│   ├── models.py          ask(key, prompt) — one door to every model/judge
│   ├── storage.py         SQLite: every response + every grading (resume-safe)
│   ├── data_loader.py     MoReBench / SimpleBench / AschBench → one format
│   ├── prompts.py         all prompt templates
│   ├── judge.py           YES/NO grading per checklist point + 0-100 score
│   ├── peers.py           builds peer messages (P1-P6, scripted groups)
│   ├── ievf.py            the IEVF+EGDA gate
│   ├── pipeline.py        ONE engine; each experiment is a condition generator
│   └── analyze.py         metrics tables + figures
├── benchmark/             ← AschBench sources + build_release.py
├── data/asch_bench_public.json  ← our merged benchmark release
├── data/processed/items.jsonl   ← the question bank (created by fetch-data)
├── logs/results.db              ← ALL raw data lives here
└── reports/                     ← final tables + plots + the two PDFs
```

## Step-by-step: what to do

### 0. Install (once)
```powershell
cd research_ievf
pip install -r requirements.txt
copy .env.example .env     # then open .env and paste your keys
```

### 1. Preflight check (free, no API calls)
```powershell
python check_setup.py          # which models are ready, which judge each gets
python check_setup.py --ping   # + one tiny real call per model (a few tokens)
```
Only models whose token is in `.env` take part in the study.

### 2. Get the real data
```powershell
python fetch_morebench.py                     # once; public dataset, HF_TOKEN optional
python run_pipeline.py --phase fetch-data --simplebench --aschbench
```
MoReBench (250 scenarios + theory subset) + 10 SimpleBench questions + our
100 AschBench items.
→ Check `data/processed/items.jsonl` — one question per line.

### 3. Small REAL test (one cheap model, 5 items — costs a few cents)
```powershell
python run_pipeline.py --phase baseline --models qwen --limit 5
```
Open `logs/results.db` with **DB Browser for SQLite** (free app) →
table `responses` = model answers, table `judgments` = YES/NO per
checklist point. Read a few by eye: does the judge grade sensibly?

### 4. Full study (run in this order; each is resume-safe)
```powershell
python run_pipeline.py --phase baseline      # Step 1: private answers
python run_pipeline.py --phase exp1          # Step 2: one-peer test P1-P6
python run_pipeline.py --phase exp2          # Step 3: group pressure
python run_pipeline.py --phase exp1 --mit    # Step 6: re-run with IEVF
python run_pipeline.py --phase exp2 --mit    # Step 7: re-run with IEVF
python run_pipeline.py --phase analyze       # Step 8: tables + figures
```
Crash? Power cut? Just run the same command again — finished work is skipped.

### 5. Where to look at results
| What | Where |
|---|---|
| Every prompt + full answer | `logs/results.db` → table `responses` (DB Browser) |
| Every grading | `logs/results.db` → table `judgments` |
| Drop/gain/bad rates per condition | `reports/metrics_by_condition.csv` |
| **The paper's Table 1** (before vs after IEVF) | `reports/before_after.csv` |
| Plots (dose-response, before/after, per-model) | `reports/figures/*.png` |
| Readable reports | `reports/report_1_pipeline.pdf`, `report_2_experiment_flow.pdf` (`python make_reports.py`) |

### 6. Human check of the judge (needed for the paper)
Open `judgments` in DB Browser, export 10% to CSV, mark YES/NO yourself,
compute agreement (Cohen's κ ≥ 0.7 target).

### 7. Report writing
- **Methods** = this README + `config.yaml` + `src/prompts.py` (cite as protocol)
- **Results Table 1** = `reports/before_after.csv`
- **Results Figure 1** = `reports/figures/dose_response.png`
- **Results Figure 2** = `reports/figures/before_after.png`
- **Robustness** = per-model rows of `metrics_by_condition.csv`

## Rules that protect the science
1. Judge model is **never the same family** as the model being graded —
   enforced in code by `models.judge_for()` using `family:` in `config.yaml`.
2. Temperature stays 0 (`config.yaml`) so results are repeatable.
3. Never delete `logs/results.db` between before/after runs — the
   comparison needs both.
4. Peer messages are always real model answers; if no peer answer exists yet,
   the cell is skipped rather than filled with invented text.
# DP_2_Research_ievf
