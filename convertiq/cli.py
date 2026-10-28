"""convertiq command line.

    python -m convertiq.cli synth --out data/synth.parquet
    python -m convertiq.cli run --data data/synth.parquet --out results
    python -m convertiq.cli run --aliccp data/ali_ccp_train.csv --nrows 2000000 --only multitask calibration
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import experiments as E
from .data import SynthConfig, generate, load_aliccp

ALL = ["multitask", "calibration", "delayed", "incremental"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="convertiq")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--per-day", type=int, default=60_000)
    r = sub.add_parser("run")
    r.add_argument("--data", help="synthetic parquet")
    r.add_argument("--aliccp", help="Ali-CCP preprocessed CSV")
    r.add_argument("--nrows", type=int, default=None)
    r.add_argument("--only", nargs="*", default=ALL, choices=ALL)
    r.add_argument("--out", default="results")
    r.add_argument("--threads", type=int, default=0)
    a = p.parse_args(argv)

    if a.cmd == "synth":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df = generate(SynthConfig(days=a.days, impressions_per_day=a.per_day))
        df.to_parquet(a.out, index=False)
        print(f"wrote {len(df):,} impressions to {a.out}; CTR {df.click.mean():.4f}, "
              f"CVR|click {df.purchase.sum() / df.click.sum():.4f}")
        return

    import pandas as pd
    if a.threads:
        torch.set_num_threads(a.threads)
    df = load_aliccp(a.aliccp, a.nrows) if a.aliccp else pd.read_parquet(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fns = {"multitask": E.multitask, "calibration": E.calibration,
           "delayed": E.delayed_feedback, "incremental": E.incremental}
    for name in a.only:
        print(f"== {name}", flush=True)
        t0 = time.time()
        res = fns[name](df)
        res["seconds"] = round(time.time() - t0, 1)
        (out / f"{name}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
