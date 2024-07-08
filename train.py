import optuna
import pytorch_lightning as pl

from data_loading import DataModule
from model import NanoFFModel


def objective(trial, platform: str, save_as: str):
    pl.seed_everything(0)
    data_module = DataModule(
        platform,
        N_train=400_000,
        N_val=1_000_000,
        batch_size=2 ** trial.suggest_int("batch_size_exp", 6, 12),
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
    model = NanoFFModel(
        from_weights=False,
        trial=trial,
        platform=save_as,
        optimizer=optimizer,
        opt_args=opt_args,
    )
    trainer = pl.Trainer(
        max_epochs=3000,
        overfit_batches=3,
        check_val_every_n_epoch=250,
        precision=32,
        logger=False,
    )
    trainer.fit(model, data_module)
    return val_loss


def generate_objective(platform: str, save_as: str):
    pl.seed_everything(0)
    return lambda trial: objective(trial, platform, save_as)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Volt_7_July2",
        direction="minimize",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=748),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(dict(
            lr=0.0014664174213001465,
            batch_size_exp=7,
            optimizer="rmsprop",
            weight_decay=0.0016544514184878858,
            alpha=0.4512242322713218,
            momentum=0.7434816089935976,
            centered=False,
        ))
    study.optimize(
        generate_objective(platform="voltlat_large", save_as="CHEVROLET_VOLT"),
        n_trials=30,
    )
