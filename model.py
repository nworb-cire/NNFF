import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn


class ManifoldConstrainedModel(nn.Module):
    def __init__(
            self,
            hidden_dims: tuple[int, int, int] = (16, 8, 4),
    ):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dims[0], 4))
        # self.b1 = nn.Parameter(torch.randn(hidden_dims[0]))
        self.W2 = nn.Parameter(torch.randn(hidden_dims[1], hidden_dims[0]))
        # self.b2 = nn.Parameter(torch.randn(hidden_dims[1]))
        self.W3 = nn.Parameter(torch.randn(hidden_dims[2], hidden_dims[1]))
        # self.b3 = nn.Parameter(torch.randn(hidden_dims[2]))
        self.W4 = nn.Parameter(torch.randn(2, hidden_dims[2]))
        # self.b4 = nn.Parameter(torch.randn(2))
        self.softplus = nn.Softplus()

    def transform_parameters(self):
        W1 = torch.concat([torch.abs(self.W1[:, 0].unsqueeze(1)), self.W1[:, 1:]], dim=1)
        W2 = -torch.abs(self.W2)
        W3 = -torch.abs(self.W3)
        W4 = torch.concat([self.W4[0].unsqueeze(0), torch.abs(self.W4[1].unsqueeze(0))], dim=0)
        return W1, W2, W3, W4

    def forward(self, x):
        """
        Forward pass constraining the output to a manifold where the following condition holds:

        ∂y₁/∂x₁ = ReLU(h₁(x)) ⊙ ReLU(h₂(x)) ⊙ ReLU(h₃(x)) ⊙ W₄[0,:] W₃ W₂ W₁[:, 0] > 0
        :param x:
        :return:
        """
        W1, W2, W3, W4 = self.transform_parameters()
        h1 = torch.relu(W1 @ x.T)
        h2 = torch.relu(W2 @ h1)
        h3 = torch.relu(W3 @ h2)
        y = W4 @ h3
        return y.T

    def serialize(self):
        W1, W2, W3, W4 = self.transform_parameters()
        return {
            "w_1": W1.detach().cpu().numpy().T.tolist(),
            # "b_1": self.b1.detach().cpu().numpy().tolist(),
            "w_2": W2.detach().cpu().numpy().T.tolist(),
            # "b_2": self.b2.detach().cpu().numpy().tolist(),
            "w_3": W3.detach().cpu().numpy().T.tolist(),
            # "b_3": self.b3.detach().cpu().numpy().tolist(),
            "w_4": W4.detach().cpu().numpy().T.tolist(),
            # "b_4": self.b4.detach().cpu().numpy().tolist(),
        }


class Sequential(nn.Module):
    def __init__(
            self,
            hidden_dims: tuple[int, int, int] = (16, 8, 4),
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 2),
        )

    def forward(self, x):
        return self.model(x)

    def serialize(self):
        return {
            "w_1": self.model[0].weight.detach().cpu().numpy().T.tolist(),
            "b_1": self.model[0].bias.detach().cpu().numpy().tolist(),
            "w_2": self.model[2].weight.detach().cpu().numpy().T.tolist(),
            "b_2": self.model[2].bias.detach().cpu().numpy().tolist(),
            "w_3": self.model[4].weight.detach().cpu().numpy().T.tolist(),
            "b_3": self.model[4].bias.detach().cpu().numpy().tolist(),
            "w_4": self.model[6].weight.detach().cpu().numpy().T.tolist(),
            "b_4": self.model[6].bias.detach().cpu().numpy().tolist(),
        }


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

        self.model = ManifoldConstrainedModel(hidden_dims)
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
        if self.trial is not None:
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            self.trial.report(val_loss, self.current_epoch)
            self.trial.set_user_attr("weights", self.serialize())
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

    def serialize(self):
        return {
            **self.model.serialize(),
            "input_norm_mat": self.input_norm_mat.detach().cpu().numpy().tolist(),
            "output_norm_mat": self.output_norm_mat.detach().cpu().numpy().tolist(),
            "temperature": self.temperature,
        }

    def on_train_end(self) -> None:
        x = np.linspace(-3, 3, 100)
        roll_ = 0.21  # median roll
        roll = roll_ * np.ones_like(x)
        v_ego_ = 10
        v_ego = v_ego_ * np.ones_like(x)
        a_ego = np.zeros_like(x)
        x = np.stack([x, roll, v_ego, a_ego], axis=1)
        y = self.forward(torch.tensor(x, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        y = 2 * y - 1

        plt.plot(x[:, 0], y[:, 0])
        plt.xlabel("Lateral Acceleration (m/s^2)")
        plt.ylabel("Steer Command")
        plt.title(f"roll = {roll_}, vEgo = {v_ego_}, aEgo = 0")
        plt.show()
        path = Path(f"logs/{self.platform}/{datetime.today().strftime('%b_%d')}/{self.trial.number}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
