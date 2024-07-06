import os
import json
import glob

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NanoFFModel(pl.LightningModule):
    def __init__(
            self,
            lr: float,
            hidden_dims: tuple[int, int, int] = (16, 8, 4),
            from_weights: bool = False,
            trial: optuna.Trial | None = None,
            platform: str | None = None,
    ):
        super().__init__()
        self.lr = lr
        self.trial = trial
        self.platform = platform
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 2),
        )
        if from_weights:
            with open("/Users/eric/PycharmProjects/openpilot/selfdrive/car/torque_data/neural_ff_weights.json",
                      "r") as f:
                bolt_weights = json.load(f)["CHEVROLET_BOLT_EUV"]
            self.model[0].weight.data = torch.tensor(bolt_weights["w_1"]).T
            self.model[0].bias.data = torch.tensor(bolt_weights["b_1"])
            self.model[2].weight.data = torch.tensor(bolt_weights["w_2"]).T
            self.model[2].bias.data = torch.tensor(bolt_weights["b_2"])
            self.model[4].weight.data = torch.tensor(bolt_weights["w_3"]).T
            self.model[4].bias.data = torch.tensor(bolt_weights["b_3"])
            self.model[6].weight.data = torch.tensor(bolt_weights["w_4"]).T
            self.model[6].bias.data = torch.tensor(bolt_weights["b_4"])

        # define constant parameters
        self.input_norm_mat = nn.Parameter(torch.tensor([[-3.0, 3.0], [-3.0, 3.0], [0.0, 40.0], [-3.0, 3.0]]), requires_grad=False)
        self.output_norm_mat = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=False)
        self.temperature = 100.0

    def loss_fn(self, y_hat, y):
        """NLL of Laplace distribution"""
        mu = y_hat[:, 0]
        theta = y_hat[:, 1]
        loss = torch.mean((torch.abs(mu - y) / torch.exp(theta)) + theta)
        return loss

    def forward(self, x):
        y = (x - self.input_norm_mat[:, 0]) / (self.input_norm_mat[:, 1] - self.input_norm_mat[:, 0])
        y = self.model(y)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.plot(self.current_epoch, self.trainer.log_dir)
        if self.trial is not None:
            self.trial.report(self.trainer.callback_metrics["val_loss"].item(), self.current_epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def plot(self, epoch: int | None = None, dir_out: str | None = None):
        lat_accel = torch.linspace(-3, 3, 100)
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        v_ego = 16
        for i, roll in enumerate([-1, 0, 1]):
            for j, a_ego in enumerate([-1, 0, 1]):
                x = torch.stack([lat_accel, torch.full_like(lat_accel, roll), torch.full_like(lat_accel, v_ego),
                                 torch.full_like(lat_accel, a_ego)], dim=1).to(self.input_norm_mat.device)
                y_hat = self.forward(x)
                y_hat = y_hat * (self.output_norm_mat[1] - self.output_norm_mat[0]) + self.output_norm_mat[0]
                steer_cmd = y_hat[:, 0].cpu().detach().numpy()
                ax[i, j].plot(lat_accel, steer_cmd)
                ax[i, j].set_title(f"roll={roll}, a_ego={a_ego}")
                ax[i, j].set_ylim(-1.3, 1.3)
        plt.tight_layout()
        if dir_out is not None:
            plt.savefig(f"{dir_out}/plot_{epoch}.png")
        else:
            plt.show()

        plt.clf()
        x = torch.stack([lat_accel, torch.zeros_like(lat_accel), torch.full_like(lat_accel, 16.), torch.zeros_like(lat_accel)], dim=1).to(self.input_norm_mat.device)
        y_hat = self.forward(x)
        theta = torch.exp(y_hat[:, 1]).cpu().detach().numpy()
        plt.plot(lat_accel, theta)
        plt.title("theta")
        plt.xlabel("lateral_accel")
        plt.ylabel("theta")
        if dir_out is not None:
            plt.savefig(f"{dir_out}/theta_{epoch}.png")
        else:
            plt.show()

    def save(self):
        assert self.platform is not None
        d = {
            "w_1": self.model[0].weight.detach().numpy().T.tolist(),
            "b_1": self.model[0].bias.detach().numpy().tolist(),
            "w_2": self.model[2].weight.detach().numpy().T.tolist(),
            "b_2": self.model[2].bias.detach().numpy().tolist(),
            "w_3": self.model[4].weight.detach().numpy().T.tolist(),
            "b_3": self.model[4].bias.detach().numpy().tolist(),
            "w_4": self.model[6].weight.detach().numpy().T.tolist(),
            "b_4": self.model[6].bias.detach().numpy().tolist(),
            "input_norm_mat": self.input_norm_mat.detach().numpy().tolist(),
            "output_norm_mat": self.output_norm_mat.detach().numpy().tolist(),
            "temperature": self.temperature,
        }

        path = "/Users/eric/PycharmProjects/openpilot/selfdrive/car/torque_data/neural_ff_weights.json"
        with open(path, "r") as f:
            existing = json.load(f)
        existing[self.platform] = d
        with open(path, "w") as f:
            f.write(json.dumps(existing))


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
        df = df[(df["steer_cmd"] >= -1) & (df["steer_cmd"] <= 1)]
        df = df[(df["v_ego"] >= 3)]
        if self.symmetrize:
            raise NotImplementedError
        dataset = self.split(df)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        assert self.df_val is not None
        df = self.df_val
        dataset = self.split(df)
        # larger batch size since we aren't computing gradients
        return DataLoader(dataset, batch_size=4 * self.batch_size, shuffle=False)


def objective(trial, platform: str, save_as: str):
    pl.seed_everything(0)
    data_module = DataModule(platform, N_train=320_000, N_val=80_000, batch_size=2 ** trial.suggest_int("batch_size_exp", 6, 10))
    model = NanoFFModel(
        lr=trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        from_weights=False,
        trial=trial,
        platform=save_as,
    )
    trainer = pl.Trainer(
        max_epochs=2000,
        overfit_batches=3,
        check_val_every_n_epoch=250,
        precision=32,
        logger=False,
    )
    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics["val_loss"].item()
    if len(trial.study.best_trials) == 0 or val_loss <= trial.study.best_value:
        print(f"New best model found with val_loss={val_loss}")
        model.save()
    return val_loss


def generate_objective(platform: str, save_as: str):
    pl.seed_everything(0)
    return lambda trial: objective(trial, platform, save_as)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Volt_Comma",
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=598),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(dict(
            lr=0.0003516815619480295,
            batch_size_exp=7,
        ))
    study.optimize(
        generate_objective(platform="CHEVROLET_VOLT_PREMIER_2017", save_as="CHEVROLET_VOLT"),
        n_trials=30,
    )
