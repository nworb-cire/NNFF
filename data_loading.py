import glob
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataModule(pl.LightningDataModule):
    df_train: pd.DataFrame
    df_val: pd.DataFrame

    def __init__(self, platform: str, N_train: int, N_val: int, symmetrize: bool = False, batch_size: int = 64):
        super().__init__()
        self.platform = platform
        self.N_train = N_train
        self.N_val = N_val
        self.symmetrize = symmetrize
        self.batch_size = batch_size

    def setup(self, stage: str | None = None):
        if os.path.isdir(f"data/{self.platform}"):
            files = glob.glob(f"data/{self.platform}/*.csv")
            assert len(files) > 0, f"No data found for {self.platform}"
            df = []
            for file in files:
                df.append(pd.read_csv(file))
            df = pd.concat(df)
        else:
            df = pd.read_feather(f"data/{self.platform}.feather")

        df = df[(df["steer_cmd"] >= -1) & (df["steer_cmd"] <= 1)]
        df = df[(df["roll"] >= -0.17) & (df["roll"] <= 0.17)]  # +/- 10 degrees
        df["roll"] = df["roll"] * 9.8

        self.df_val = df.sample(self.N_val, random_state=0)
        self.df_train = df.drop(self.df_val.index).sample(self.N_train, random_state=0)

    @staticmethod
    def split(df: pd.DataFrame) -> TensorDataset:
        x_cols = [
            "lateral_accel",
            "roll",
            "v_ego",
            "a_ego",
        ]
        y_col = "steer_cmd"
        x = torch.tensor(df[x_cols].values, dtype=torch.float32)
        y = torch.tensor(df[y_col].values, dtype=torch.float32)
        y = (y + 1) / 2  # normalize to [0, 1]
        return TensorDataset(x, y)

    def train_dataloader(self):
        assert self.df_train is not None
        df = self.df_train
        if self.symmetrize:
            raise NotImplementedError
        dataset = self.split(df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        assert self.df_val is not None
        df = self.df_val
        df = df[(df["v_ego"] >= 3)]
        dataset = self.split(df)
        # larger batch size since we aren't computing gradients
        return DataLoader(dataset, batch_size=4 * self.batch_size, shuffle=False)
