import glob
import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

from validation.wrapper import ValidationModule


def read_csv(file, columns: list[str]):
    df = pd.read_csv(file)
    if not df.latActive.all() or df.steeringPressed.any():
        return None
    return df[columns]


def load_data(columns: list[str]):
    files = glob.glob("data/CHEVROLET_VOLT_PREMIER_2017/1*.csv")
    with joblib.Parallel(n_jobs=-1) as parallel:
        dfs = parallel(joblib.delayed(lambda file: read_csv(file, columns))(file) for file in files)
    none_count = sum(1 for df in dfs if df is None)
    dfs = [df for df in dfs if df is not None]
    print(f"Loaded {len(dfs)} files with total of {sum(len(df) for df in dfs):,} samples")
    print(f"Discarded {none_count} files")
    return dfs


if __name__ == "__main__":
    if os.path.isfile("model.pkl"):
        model = joblib.load("model.pkl")
        pretrained = True
    else:
        model = XGBRegressor(n_estimators=100, max_depth=15)
        pretrained = False
    wrapper = ValidationModule(
        model=model,
        ar_order=(30, 30),
        target="latAccelLocalizer",
        exog=["steerFiltered", "roll", "vEgo", "aEgo"],
        actuator_command=None,
    )
    dfs = load_data([wrapper.target] + wrapper.exog)
    dfs_train = dfs[:-200]
    dfs_val = dfs[-200:]
    if not pretrained:
        wrapper.fit(dfs_train)
        joblib.dump(wrapper.model, "model.pkl")

    with joblib.Parallel(n_jobs=-1) as parallel:
        preds = parallel(joblib.delayed(wrapper.predict)(df) for df in dfs_val)
    for i, (df, pred) in enumerate(zip(dfs_val, preds)):
        if i % 20 != 0:
            continue
        ground_truth = df[wrapper.target][:wrapper.ar_order[0] + 20]
        plt.plot(ground_truth, label="Ground truth")
        plt.plot(pred, label="Prediction")
        plt.title(f"Validation sample {i}")
        plt.vlines(wrapper.ar_order[0], min(ground_truth.min(), pred.min()), max(ground_truth.max(), pred.max()), colors="k", linestyles="-")
        plt.xlabel("Timestep")
        plt.ylabel("Lateral Acceleration")
        plt.legend()
        plt.show()
    losses = [
        np.nanmean(np.abs(df[wrapper.target] - pred) ** 2)
        for df, pred in zip(dfs_val, preds)
    ]
    print(f"Validation loss: {np.mean(losses):.4f}")
