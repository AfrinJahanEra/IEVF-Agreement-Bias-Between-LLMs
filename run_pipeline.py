"""One command runs everything.

  # 0. check config + tokens before spending anything
  python check_setup.py

  # 1. build the question bank (MoReBench + SimpleBench + AschBench)
  python run_pipeline.py --phase fetch-data --simplebench --aschbench

  # 2. small real test: one model, 5 items
  python run_pipeline.py --phase baseline --models qwen --limit 5

  # 3. full study
  python run_pipeline.py --phase baseline
  python run_pipeline.py --phase exp1
  python run_pipeline.py --phase exp2
  python run_pipeline.py --phase analyze
  python run_pipeline.py --phase exp1 --mit
  python run_pipeline.py --phase exp2 --mit
  python run_pipeline.py --phase analyze

Which LLMs run is decided by the tokens in .env: a model configured in
config.yaml without its token is skipped automatically.
"""
import argparse

from src import analyze, data_loader, models, pipeline, storage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["fetch-data", "baseline", "exp1", "exp2",
                             "analyze"])
    ap.add_argument("--models", default=None,
                    help="comma list, e.g. qwen,deepseek "
                         "(default: every model whose token is in .env)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N items (for quick tests)")
    ap.add_argument("--mit", action="store_true",
                    help="run the phase with IEVF+EGDA switched on")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake answers, no real API calls - writes to a "
                         "separate logs/dryrun_results.db, never the real "
                         "one. Free; tests all plumbing.")
    ap.add_argument("--morebench-n", type=int, default=250,
                    help="how many MoReBench items to sample (fetch-data)")
    ap.add_argument("--simplebench", action="store_true",
                    help="fetch-data: also append the 10 SimpleBench items")
    ap.add_argument("--aschbench", action="store_true",
                    help="fetch-data: also append our 100 AschBench items")
    args = ap.parse_args()

    if args.dry_run:
        models.DRY_RUN = True
        storage.DRY_RUN = True
        print("[dry-run] fake answers, no API calls, writing to "
              "logs/dryrun_results.db (the real results.db is untouched)")

    if args.phase == "fetch-data":
        items = data_loader.load_morebench(n=args.morebench_n,
                                           include_theory=True)
        if args.simplebench:
            items += data_loader.load_simplebench()
        if args.aschbench:
            items += data_loader.load_aschbench()
        data_loader.save_items(items)
        return

    if args.phase == "analyze":
        storage.init_db()
        analyze.run_analysis()
        return

    items = data_loader.load_items()
    only = args.models.split(",") if args.models else None
    model_keys = models.study_models(only)
    pipeline.run(items, model_keys, args.phase, limit=args.limit,
                 mit=args.mit)


if __name__ == "__main__":
    main()
