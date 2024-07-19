import argparse
import json
from datetime import datetime

import optuna
import pytorch_lightning as pl

from data_loading import CommaData
from model import NanoFFModel


def objective(trial, platform: str, save_as: str):
    pl.seed_everything(0)
    data = CommaData(
        platform,
        batch_size=2 ** 16,
    )
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adamw"])
    opt_args = {
        "lr": trial.suggest_float("lr", 1e-6, 1., log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-2),
    }
    match optimizer:
        case "adam":
            opt_args["amsgrad"] = trial.suggest_categorical("amsgrad", [True, False])
        case "sgd":
            opt_args["momentum"] = trial.suggest_float("momentum", 0.0, 1.0)
            opt_args["nesterov"] = trial.suggest_categorical("nesterov", [True, False])
            if not opt_args["nesterov"]:
                opt_args["dampening"] = trial.suggest_float("dampening", 0.0, 1.0)
        case "rmsprop":
            opt_args["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
            opt_args["momentum"] = trial.suggest_float("momentum", 0.0, 1.0)
            opt_args["centered"] = trial.suggest_categorical("centered", [True, False])
        case "adamw":
            opt_args["amsgrad"] = trial.suggest_categorical("amsgrad", [True, False])
    model = NanoFFModel(
        from_weights=False,
        trial=trial,
        platform=save_as,
        optimizer=optimizer,
        opt_args=opt_args,
    )
    trainer = pl.Trainer(
        max_epochs=data.N_epochs,
        overfit_batches=3,
        check_val_every_n_epoch=data.N_epochs // 10,
        precision=32,
        logger=False,
    )
    trainer.fit(model, datamodule=data)
    try:
        loss = trainer.callback_metrics["val_loss"].item()
    except KeyError:
        loss = float("nan")

    try:
        if loss < trial.study.best_trial.value:
            with open("best_params.json", "w") as f:
                best_params = json.load(f)
                best_params[platform] = trial.params
                best_params[platform]["loss"] = loss
                json.dump(best_params, f)
            with open("/Users/eric/PycharmProjects/openpilot/selfdrive/car/torque_data/neural_ff_weights.json", "w") as f:
                model_params = json.load(f)
                model_params[save_as] = model.serialize()
                json.dump(model_params, f)
    except ValueError:
        pass
    return loss


def generate_objective(platform: str, save_as: str):
    return lambda trial: objective(trial, platform, save_as)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str)
    args = parser.parse_args()

    platform = args.platform
    with open("fingerprint_migration.json") as f:
        fingerprint_migration = json.load(f)
    assert platform in fingerprint_migration, f"Platform {platform} not found in fingerprint_migration.json"

    study = optuna.create_study(
        study_name=f"{platform}_{datetime.today().strftime("%b_%d")}",
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=500),
    )
    if len(study.trials) == 0:
        with open("best_params.json") as f:
            best_params = json.load(f)
        if platform in best_params:
            best_params[platform].pop("loss")
            study.enqueue_trial(best_params[platform])
    study.optimize(
        generate_objective(platform=fingerprint_migration[platform], save_as=platform),
        n_trials=50 - len(study.trials),
    )
