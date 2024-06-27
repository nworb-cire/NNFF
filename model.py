import json

import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NanoFFModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        # define constant parameters
        self.input_norm_mat = nn.Parameter(torch.tensor([[-3.0, 3.0], [-3.0, 3.0], [0.0, 40.0], [-3.0, 3.0]]), requires_grad=False)
        self.output_norm_mat = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=False)
        self.temperature = 100.0

    def loss_fn(self, y_hat, y):
        """NLL of Laplace distribution"""
        loc = y_hat[:, 0]
        scale = y_hat[:, 1] / self.temperature
        loss = torch.mean(torch.abs(loc - y) / torch.exp(scale) + scale)
        return loss

    def forward(self, x):
        y = (x - self.input_norm_mat[:, 0]) / (self.input_norm_mat[:, 1] - self.input_norm_mat[:, 0])
        y = self.model(y)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        # penalize output outside of [0, 1]
        loss += 1e-2 * torch.mean(torch.relu(-y_hat) + torch.relu(y_hat - 1))

        # penalize asymmetric distribution of output
        lat_accel = torch.linspace(-3, 0, 100)
        x = torch.stack([lat_accel, torch.zeros_like(lat_accel), torch.full_like(lat_accel, 16), torch.zeros_like(lat_accel)], dim=1).to(self.input_norm_mat.device)
        y_hat_neg = self.forward(x)
        lat_accel = torch.linspace(0, 3, 100)
        x = torch.stack([lat_accel, torch.zeros_like(lat_accel), torch.full_like(lat_accel, 16), torch.zeros_like(lat_accel)], dim=1).to(self.input_norm_mat.device)
        y_hat_pos = self.forward(x)
        loss += 1e-2 * torch.mean(torch.abs(y_hat_neg - y_hat_pos))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.plot(self.current_epoch, self.trainer.log_dir)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
                ax[i, j].set_ylim(-1.1, 1.1)
        plt.tight_layout()
        if dir_out is not None:
            plt.savefig(f"{dir_out}/plot_{epoch}.png")
        else:
            plt.show()

    def save(self):
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
        existing["CHEVROLET_VOLT"] = d
        with open(path, "w") as f:
            f.write(json.dumps(existing))


def get_dataset(platform: str, size: int = -1) -> TensorDataset:
    df = pd.read_parquet(f"data/{platform}.parquet")
    # remove outliers
    df = df[(df["steer_cmd"] > -1) & (df["steer_cmd"] < 1)]

    if size > 0:
        df = df.sample(size)
    x_cols = [
        "lateral_accel",
        "roll",
        "v_ego",
        "a_ego",
    ]
    y_col = "steer_cmd"
    x = df[x_cols].values
    y = df[y_col].values
    y = (y + 1) / 2  # normalize to [0, 1]

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


if __name__ == "__main__":
    pl.seed_everything(0)
    N = 400_000
    dataset = get_dataset("voltlat_large", size=N)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * N), int(0.2 * N)])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    model = NanoFFModel()
    trainer = pl.Trainer(
        max_epochs=1000,
        overfit_batches=3,
        check_val_every_n_epoch=100,
        gradient_clip_val=2.0,
    )
    trainer.fit(model, train_loader, val_loader)
    model.save()
