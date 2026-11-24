"""Run the experiments and write results/summary.md.

    python -m lob.run                                  # all models, k = 10, 50, 100
    python -m lob.run --models deeplob --horizons 50   # one model, one horizon
    python -m lob.run --summary-only                   # rebuild summary.md from saved runs

Each (model, horizon) result is saved to results/runs/ as soon as it finishes, so an
interrupted run can resume where it stopped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from .data import HORIZONS, WINDOW, download, load
from .models import build, n_params
from .train import Windows, baselines, fit, latency_us, metrics, predict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"


def run_one(name: str, k: int, train, val, test, args, log) -> dict:
    h = HORIZONS.index(k)
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{name}_k{k}.json"
    if path.exists():
        return json.loads(path.read_text())
    log.info("%s, k=%d", name, k)
    if name == "baselines":
        res = baselines(train, test, h, seed=args.seed)
    else:
        torch.manual_seed(args.seed)
        model = build(name)
        info = fit(model, train, val, h, epochs=args.epochs, samples_per_epoch=args.samples,
                   seed=args.seed, log=lambda m: log.info(m))
        stats = info.pop("stats")
        y, p = predict(model, Windows(test, h, stats=stats))
        res = {name: {**metrics(y, p), **info, "params": n_params(model), **latency_us(model)}}
        torch.save(model.state_dict(), RUNS / f"{name}_k{k}.pt")
    path.write_text(json.dumps(res, indent=1))
    return res


def summarize(test_labels) -> None:
    rows = []
    for f in sorted(RUNS.glob("*.json")):
        k = int(f.stem.split("_k")[1])
        for model, m in json.loads(f.read_text()).items():
            rows.append({"horizon_k": k, "model": model, "accuracy": m["accuracy"],
                         "f1_macro": m["f1_macro"], "precision": m["precision_macro"],
                         "recall": m["recall_macro"],
                         "latency_p50_us": m.get("p50_us", m.get("latency_p50_us")),
                         "params": m.get("params")})
    df = pd.DataFrame(rows).sort_values(["horizon_k", "f1_macro"], ascending=[True, False])
    df.to_csv(RESULTS / "results.csv", index=False)
    share = {k: (test_labels[WINDOW - 1:, HORIZONS.index(k)] == 1).mean() for k in HORIZONS}
    lines = ["# Results", "",
             "FI-2010, no-auction, decimal-precision normalised. Trained on days 1-7 (last 20% "
             "held out for early stopping), tested on days 8-10. Macro F1 over up / stationary / down.",
             "",
             "Share of 'stationary' labels in the test set by horizon: "
             + ", ".join(f"k={k}: {v:.0%}" for k, v in share.items()), "",
             df.to_markdown(index=False, floatfmt=".4f"), ""]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print(df.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--models", nargs="+", default=["baselines", "deeplob", "transformer"])
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--samples", type=int, default=32768, help="training windows per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("lob")
    download(args.data)
    train, val, test = load(args.data)
    log.info("train %d, val %d, test %d events", len(train.lob), len(val.lob), len(test.lob))
    if not args.summary_only:
        for name in args.models:
            for k in args.horizons:
                run_one(name, k, train, val, test, args, log)
    summarize(test.labels)


if __name__ == "__main__":
    main()
