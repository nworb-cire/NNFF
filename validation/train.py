import glob
import os

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from validation.models import TrivialModel
from validation.wrapper import ValidationModule


def load_data(columns: list[str]):
    files = glob.glob("data/CHEVROLET_VOLT_PREMIER_2017/1*.csv")
    with joblib.Parallel(n_jobs=-1) as parallel:
        dfs = parallel(joblib.delayed(pd.read_csv)(file) for file in files)
    dfs = [df[columns] for df in dfs]  # TODO: merge this with the delayed call above
    print(f"Loaded {len(dfs)} files with total of {sum(len(df) for df in dfs):,} samples")
    return dfs


if __name__ == "__main__":
    if os.path.isfile("model.pkl"):
        model = joblib.load("model.pkl")
        pretrained = True
    else:
        model = TrivialModel("latAccelLocalizer")
        pretrained = False
    wrapper = ValidationModule(
        model=model,
        ar_order=(12, 12),
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
        ground_truth = df[wrapper.target][:12+15]
        plt.plot(ground_truth, label="Ground truth")
        plt.plot(pred, label="Prediction")
        plt.title(f"Validation sample {i}")
        plt.legend()
        plt.show()
    losses = [
        ((df[wrapper.target][12:] - pred) ** 2).mean()
        for df, pred in zip(dfs_val, preds)
    ]
    print(f"Validation loss: {np.mean(losses):.4f}")
