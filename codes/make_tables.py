"""Re-creates Table 2 and Table 3 of the Isolation Forest paper as CSV files in results/.

    python make_tables.py                                   # everything (the three big datasets take about 2 hours for ORCA)
    python make_tables.py --methods iforest lof             # only some methods
    python make_tables.py --skip Http ForestCover Mulcross  # leave out the slow datasets
    python make_tables.py --only Pima Breastw --runs 3      # quick test

Output (results/): table2.csv, iforest.csv, orca.csv, lof.csv, rf.csv (ours vs. paper, per method),
table3_reproduced.csv and table3_paper.csv (the combined layout of the paper's Table 3).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets import DEFS, PAPER, load_all, table2
from iforest import run_iforest
from orca import run_orca
from lof import run_lof
from rf import run_rf

LOF_SETS = {n for n, p in PAPER.items() if not np.isnan(p["lof"][0])}
RF_SETS = {n for n, p in PAPER.items() if not np.isnan(p["rf"][0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--methods", nargs="+", default=["iforest", "orca", "lof", "rf"],
                    choices=["iforest", "orca", "lof", "rf"])
    ap.add_argument("--only", nargs="+"); ap.add_argument("--skip", nargs="+", default=[])
    ap.add_argument("--runs", type=int, default=10, help="runs averaged for iForest and RF")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    data = {n: v for n, v in load_all(a.data, a.only).items() if n not in a.skip}
    table2(data).to_csv(out / "table2.csv", index=False)

    res = {m: [] for m in a.methods}
    for name, (X, y) in data.items():
        p = PAPER[name]
        if "iforest" in a.methods:
            print(f"[iForest] {name}", flush=True)
            aucs, tr, ev = run_iforest(X, y, runs=a.runs)
            res["iforest"].append(dict(dataset=name, auc=np.mean(aucs), sd=np.std(aucs), paper_auc=p["iforest"][0],
                                       train_s=tr, eval_s=ev, total_s=tr + ev, paper_total_s=p["iforest"][3]))
        if "orca" in a.methods:
            print(f"[ORCA] {name}", flush=True)
            s, sec = run_orca(X)
            res["orca"].append(dict(dataset=name, auc=roc_auc_score(y, s), paper_auc=p["orca"][0],
                                    time_s=sec, paper_time_s=p["orca"][1]))
        if "lof" in a.methods and name in LOF_SETS:
            print(f"[LOF] {name}", flush=True)
            s, sec = run_lof(X)
            res["lof"].append(dict(dataset=name, auc=roc_auc_score(y, s), paper_auc=p["lof"][0],
                                   time_s=sec, paper_time_s=p["lof"][1]))
        if "rf" in a.methods and name in RF_SETS:
            print(f"[RF] {name}", flush=True)
            aucs, sec = run_rf(X, y, runs=a.runs)
            res["rf"].append(dict(dataset=name, auc=np.mean(aucs), sd=np.std(aucs), paper_auc=p["rf"][0],
                                  time_s=sec, paper_time_s=p["rf"][1]))
        for m, rows in res.items():                                   # save after every dataset
            pd.DataFrame(rows).round(3).to_csv(out / f"{m}.csv", index=False)

    # combined Table 3 in the layout of the paper
    ours, paper = [], []
    for name in data:
        r, q = {"dataset": name}, {"dataset": name}
        for m, label in (("iforest", "iForest"), ("orca", "ORCA"), ("lof", "LOF"), ("rf", "RF")):
            row = next((x for x in res.get(m, []) if x["dataset"] == name), None)
            r[f"AUC {label}"] = round(row["auc"], 3) if row else np.nan
            q[f"AUC {label}"] = PAPER[name][m][0]
        if "iforest" in res:
            row = next(x for x in res["iforest"] if x["dataset"] == name)
            for key, lab in (("train_s", "train"), ("eval_s", "eval"), ("total_s", "total")):
                r[f"time iForest {lab}"] = round(row[key], 2)
            q.update({"time iForest train": PAPER[name]["iforest"][1], "time iForest eval": PAPER[name]["iforest"][2],
                      "time iForest total": PAPER[name]["iforest"][3]})
        for m, label in (("orca", "ORCA"), ("lof", "LOF"), ("rf", "RF")):
            row = next((x for x in res.get(m, []) if x["dataset"] == name), None)
            r[f"time {label}"] = round(row["time_s"], 2) if row else np.nan
            q[f"time {label}"] = PAPER[name][m][1]
        ours.append(r); paper.append(q)
    pd.DataFrame(ours).to_csv(out / "table3_reproduced.csv", index=False)
    pd.DataFrame(paper).to_csv(out / "table3_paper.csv", index=False)
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
