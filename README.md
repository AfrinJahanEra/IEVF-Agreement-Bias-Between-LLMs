# IEVF — Agreement Bias Between LLMs

Does one LLM change another LLM's mind under peer pressure — and can the
**IEVF** framework (with the **EGDA** auditor) stop it?

**Benchmarks:** MoReBench (moral dilemmas + expert checklists) and
SimpleBench (10 verifiable probe questions). **No training, no LoRA** — pure
inference-time study via APIs.

## Project map

```
research_ievf/
├── run_pipeline.py        ← the only file you run
├── config.yaml            ← models, sizes, settings (edit this)
├── .env                   ← your API keys (create from .env.example)
├── src/
│   ├── config.py          loads config + keys
│   ├── models.py          ask(model, prompt) — one door to all 6 models
│   ├── storage.py         SQLite: every response + every grading (resume-safe)
│   ├── data_loader.py     MoReBench / SimpleBench / offline sample
│   ├── prompts.py         all prompt templates
│   ├── judge.py           YES/NO grading per checklist point + 0-100 score
│   ├── peers.py           builds peer messages (P1-P6, scripted groups)
│   ├── ievf.py            the IEVF+EGDA gate
│   ├── pipeline.py        the experiment loops (baseline, exp1, exp2)
│   └── analyze.py         metrics tables + figures
├── data/processed/items.jsonl   ← the question bank (created by fetch-data)
├── logs/results.db              ← ALL raw data lives here
└── reports/                     ← final tables + plots live here
```

## Step-by-step: what to do

### 0. Install (once)
```powershell
cd research_ievf
pip install -r requirements.txt
copy .env.example .env     # then open .env and paste your keys
```

### 1. Test with ZERO cost (no keys needed)
```powershell
python run_pipeline.py --phase fetch-data --sample
python run_pipeline.py --phase baseline --dry-run
python run_pipeline.py --phase exp1 --dry-run
```
If progress bars move and `logs/results.db` grows, the machine works.

### 2. Get the real data
```powershell
python run_pipeline.py --phase fetch-data --simplebench
```
Downloads MoReBench from Hugging Face (250 scenarios + theory subset)
and appends the 10 SimpleBench questions.
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
| Plots (dose-response, before/after) | `reports/figures/*.png` |

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
1. Judge model is **never the same family** as the model being graded
   (set `JUDGE_MODEL` in `.env`; `families:` in config.yaml shows why).
2. Temperature stays 0 (`config.yaml`) so results are repeatable.
3. Never delete `logs/results.db` between before/after runs — the
   comparison needs both.
4. `--dry-run` output is plumbing-test only; never use it in the paper.
# DP_2_Research_ievf
