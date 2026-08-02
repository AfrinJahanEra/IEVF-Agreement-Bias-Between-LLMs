"""One command runs everything. Examples:

  # 1. offline test with 5 fake items, no API keys needed:
  python run_pipeline.py --phase fetch-data --sample
  python run_pipeline.py --phase baseline --dry-run

  # 2. real data download:
  python run_pipeline.py --phase fetch-data --simplebench

  # 3. real small test on one model, 5 items:
  python run_pipeline.py --phase baseline --models qwen --limit 5

  # 4. full study:
  python run_pipeline.py --phase baseline
  python run_pipeline.py --phase exp1
  python run_pipeline.py --phase exp2
  python run_pipeline.py --phase exp1 --mit
  python run_pipeline.py --phase exp2 --mit
  python run_pipeline.py --phase analyze
"""
import argparse

from src import analyze, data_loader, pipeline, storage
from src.config import CFG, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["fetch-data", "baseline", "exp1", "exp2",
                             "analyze"])
    ap.add_argument("--models", default=None,
                    help="comma list, e.g. qwen,deepseek (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N items (for quick tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no API calls at all; fake answers, judge says YES")
    ap.add_argument("--mit", action="store_true",
                    help="run the phase with IEVF+EGDA switched on")
    ap.add_argument("--sample", action="store_true",
                    help="fetch-data: write 5 offline sample items")
    ap.add_argument("--simplebench", action="store_true",
                    help="fetch-data: also append the 10 SimpleBench items")
    ap.add_argument("--aschbench", action="store_true",
                    help="fetch-data: also append our AschBench items")
    args = ap.parse_args()

    if args.phase == "fetch-data":
        if args.sample:
            items = data_loader.sample_items()
        else:
            items = data_loader.load_morebench(n=250, include_theory=True)
        if args.simplebench:
            items += data_loader.load_simplebench()
        if args.aschbench:
            items += data_loader.load_aschbench()
        data_loader.save_items(items)
        return

    if args.phase == "analyze":
        storage.init_db()
        analyze.run_analysis(env("JUDGE_MODEL", "deepseek-chat"))
        return

    items = data_loader.load_items()
    models = args.models.split(",") if args.models else list(CFG["models"])
    unknown = [m for m in models if m not in CFG["models"]]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}. "
                         f"Known: {list(CFG['models'])}")
    pipeline.run(items, models, args.phase, mock=args.dry_run,
                 limit=args.limit, mit=args.mit)


if __name__ == "__main__":
    main()
