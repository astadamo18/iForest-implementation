"""Dataset loading, anomaly definitions (paper Table 2) and the published values (paper Table 3)."""
from pathlib import Path
import numpy as np
import pandas as pd

# order and anomaly definitions of Table 2
DEFS = {
    "Http": "attack", "ForestCover": "class 4 vs. class 2", "Mulcross": "2 clusters", "Smtp": "attack",
    "Shuttle": "classes 2, 3, 5, 6, 7", "Mammography": "class 1", "Annthyroid": "classes 1, 2",
    "Satellite": "3 smallest classes", "Pima": "pos", "Breastw": "malignant",
    "Arrhythmia": "classes 03, 04, 05, 07, 08, 09, 14, 15", "Ionosphere": "bad"}

NAN = float("nan")
# Table 3 of the paper.  iForest: (AUC, train s, eval s, total s) | ORCA, LOF, RF: (AUC, time s); NaN = NA in the paper
PAPER = {
    "Http":        dict(iforest=(1.00, 0.25, 15.33, 15.58), orca=(0.36, 9487.47), lof=(NAN, NAN), rf=(NAN, NAN)),
    "ForestCover": dict(iforest=(0.88, 0.76, 15.57, 16.33), orca=(0.83, 6995.17), lof=(NAN, NAN), rf=(NAN, NAN)),
    "Mulcross":    dict(iforest=(0.97, 0.26, 12.26, 12.52), orca=(0.33, 2512.20), lof=(NAN, NAN), rf=(NAN, NAN)),
    "Smtp":        dict(iforest=(0.88, 0.14, 2.58, 2.72),   orca=(0.80, 267.45),  lof=(NAN, NAN), rf=(NAN, NAN)),
    "Shuttle":     dict(iforest=(1.00, 0.30, 2.83, 3.13),   orca=(0.60, 156.66),  lof=(0.55, 7489.74), rf=(NAN, NAN)),
    "Mammography": dict(iforest=(0.86, 0.16, 0.50, 0.66),   orca=(0.77, 4.49),    lof=(0.67, 14647.00), rf=(NAN, NAN)),
    "Annthyroid":  dict(iforest=(0.82, 0.15, 0.36, 0.51),   orca=(0.68, 2.32),    lof=(0.72, 72.02), rf=(NAN, NAN)),
    "Satellite":   dict(iforest=(0.71, 0.46, 1.17, 1.63),   orca=(0.65, 8.51),    lof=(0.52, 217.39), rf=(NAN, NAN)),
    "Pima":        dict(iforest=(0.67, 0.17, 0.11, 0.28),   orca=(0.71, 0.06),    lof=(0.49, 1.14), rf=(0.65, 4.98)),
    "Breastw":     dict(iforest=(0.99, 0.17, 0.11, 0.28),   orca=(0.98, 0.04),    lof=(0.37, 1.77), rf=(0.97, 3.10)),
    "Arrhythmia":  dict(iforest=(0.80, 2.12, 0.86, 2.98),   orca=(0.78, 0.49),    lof=(0.73, 6.35), rf=(0.60, 2.32)),
    "Ionosphere":  dict(iforest=(0.85, 0.33, 0.15, 0.48),   orca=(0.92, 0.04),    lof=(0.89, 0.64), rf=(0.85, 0.83)),
}


def load_all(data_dir="data", only=None):
    """name -> (X, y); y = 1 for anomalies. Rows keep their original file order (ORCA depends on it)."""
    data = {}
    for name in DEFS:
        if only and name not in only:
            continue
        df = pd.read_csv(Path(data_dir) / f"{name}.csv")
        data[name] = (df.drop(columns="y").to_numpy(dtype=float), df["y"].to_numpy().astype(int))
    return data


def table2(data):
    return pd.DataFrame([{"dataset": n, "n": len(y), "d": X.shape[1], "anomaly definition": DEFS[n],
                          "proportion of anomalies": f"{100 * y.mean():.2f}%"} for n, (X, y) in data.items()])
