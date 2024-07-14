import abc
import glob
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class LateralData(pl.LightningDataModule, abc.ABC):
    df_train: pd.DataFrame
    df_val: pd.DataFrame

    x_cols: list[str]
    y_col: str

    N_epochs: int

    def __init__(self, platform: str, symmetrize: bool = False, batch_size: int = 64):
        super().__init__()
        self.platform = platform
        self.symmetrize = symmetrize
        self.batch_size = batch_size

    def symmetrize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df, df.assign(**{
            self.x_cols[1]: -df[self.x_cols[1]],
            self.x_cols[2]: -df[self.x_cols[2]],
            self.y_col: -df[self.y_col],
        })])

    def split(self, df: pd.DataFrame) -> TensorDataset:
        x = torch.tensor(df[self.x_cols].values, dtype=torch.float32)
        y = torch.tensor(df[self.y_col].values, dtype=torch.float32)
        y = (y + 1) / 2  # normalize to [0, 1]
        return TensorDataset(x, y)

    def train_dataloader(self):
        assert self.df_train is not None
        df = self.df_train
        if self.symmetrize:
            df = self.symmetrize_frame(df)
        dataset = self.split(df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        assert self.df_val is not None
        df = self.df_val
        df = df[(df[self.x_cols[2]] >= 3)]
        if self.symmetrize:
            df = self.symmetrize_frame(df)
        dataset = self.split(df)
        # larger batch size since we aren't computing gradients
        return DataLoader(dataset, batch_size=16 * self.batch_size, shuffle=False, drop_last=True)


class CommaData(LateralData):
    x_cols = [
        "latAccelSteeringAngle",
        "roll",
        "vEgo",
        "aEgo",
    ]
    y_col = "steerFiltered"

    N_epochs = 1500

    def setup(self, stage: str | None = None):
        files = glob.glob(f"data/{self.platform}/*.csv")
        assert len(files) > 0, f"No data found for {self.platform}"
        dfs = []
        for file in files:
            df = pd.read_csv(file)

            df["routeId"] = os.path.basename(file).split("_")[0]
            df[self.x_cols[0]] = df[self.x_cols[0]].shift(-8)  # actuator delay
            df[self.y_col] = -df[self.y_col]
            df[self.x_cols[1]] = df[self.x_cols[1]] * 9.8  # roll

            # filter rows where steering has not been pressed for 10 time steps
            pd.set_option("future.no_silent_downcasting", True)
            for i in range(10):
                df[f"steeringPressed{i}"] = df["steeringPressed"].shift(i)
                df[f"steeringPressed{i}"] = df[f"steeringPressed{i}"].fillna(True)
            df = df[~df[[f"steeringPressed{i}" for i in range(10)]].any(axis=1)]

            df.dropna(inplace=True, subset=self.x_cols + [self.y_col])
            dfs.append(df)
        df = pd.concat(dfs)

        # 50% holdout
        route_ids = df["routeId"].unique()
        np.random.shuffle(route_ids)
        route_ids_train = route_ids[:len(route_ids) // 2]

        self.df_train = df[df["routeId"].isin(route_ids_train)]
        self.df_val = df[~df["routeId"].isin(route_ids_train)]


class TWilsonData(LateralData):
    x_cols = [
        "lateral_accel",
        "roll",
        "v_ego",
        "a_ego",
    ]
    y_col = "steer_cmd"
    N_train = 400_000
    N_val = 1_000_000

    N_epochs = 5000

    def bucket(self, df: pd.DataFrame, bins: int | np.ndarray = 15, bucket_size: int = -1) -> pd.DataFrame:
        """Splits the DataFrame into buckets to ensure that the validation set has a representative distribution."""
        df = df.copy()
        df["steer_bucket"] = pd.cut(df[self.y_col], bins=bins, labels=False)
        if bucket_size < 0:
            bucket_size = df.groupby("steer_bucket").size().min()
        print(f"Bucket size: {bucket_size}")
        df = df.groupby("steer_bucket").apply(
            lambda x: x.sample(bucket_size, random_state=0, replace=True)
        )
        df.index = df.index.droplevel(0)
        df = df.drop(columns=["steer_bucket"])
        return df

    def setup(self, stage: str | None = None):
        df = pd.read_feather(f"data/{self.platform}.feather", columns=self.x_cols + [self.y_col])

        df = df[(df[self.y_col] >= -1) & (df[self.y_col] <= 1)]
        # roll
        df = df[(df[self.x_cols[1]] >= -0.17) & (df[self.x_cols[1]] <= 0.17)]  # +/- 10 degrees
        df[self.x_cols[1]] = df[self.x_cols[1]] * 9.8

        self.df_val = self.bucket(df)
        if self.N_val >= len(self.df_val):
            print(f"Warning: N_val ({self.N_val}) >= len(df_val) ({len(self.df_val)}), using all validation data")
        else:
            self.df_val = self.df_val.sample(self.N_val, random_state=0)
        self.df_train = self.bucket(
            df.drop(self.df_val.index),
            bins=np.concatenate([
                np.linspace(-1, -0.2, 9, endpoint=False),
                np.linspace(-0.2, 0.2, 10, endpoint=False),
                np.linspace(0.2, 1, 10, endpoint=True),
            ])
        ).sample(self.N_train, random_state=0, replace=True)
