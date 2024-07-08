import json
from typing import Literal

import optuna
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn


class NanoFFModel(pl.LightningModule):
    def __init__(
            self,
            hidden_dims: tuple[int, int, int] = (16, 8, 4),
            from_weights: bool = False,
            trial: optuna.Trial | None = None,
            platform: str | None = None,
            optimizer: Literal["adam", "sgd", "rmsprop", "adamw"] = "adam",
            opt_args: dict = {},
    ):
        super().__init__()
        self.trial = trial
        self.platform = platform
        self.optimizer = optimizer
        self.opt_args = opt_args

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
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            self.trial.report(val_loss, self.current_epoch)
            if len(self.trial.study.best_trials) == 0 or val_loss <= self.trial.study.best_value:
                print(f"New best model found with val_loss={val_loss}")
                self.save()
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def configure_optimizers(self):
        match self.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), **self.opt_args)
            case "sgd":
                optimizer = torch.optim.SGD(self.parameters(), **self.opt_args)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), **self.opt_args)
            case "adamw":
                optimizer = torch.optim.AdamW(self.parameters(), **self.opt_args)
            case _:
                raise ValueError(f"Invalid optimizer: {self.optimizer}")
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
