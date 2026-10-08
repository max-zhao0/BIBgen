from typing import Callable
import json

import torch
import numpy as np

from BIBgen import models as model_classes
from BIBgen.training.dataloaders import BaseDataLoader
from BIBgen.training.core import train, evaluate

def load_empty_model(model_config_path : str, n_timesteps : int):
    assert model_config_path.endswith(".json")
    with open(model_config_path, "r") as fin:
        model_config = json.load(fin)
    model = getattr(model_classes, model_config["name"])(n_timesteps=n_timesteps, **model_config["hyperparameters"])
    return model

def hyperoptimize(
    model_template : dict,
    training_loader : BaseDataLoader,
    validation_loader : BaseDataLoader,
    loss_fn : Callable,
    device : torch.device,
    n_timesteps : int,
    nepochs : int = 20,
    ntrials : int = 40,
):
    import optuna

    Model = getattr(model_classes, model_template["name"])
    hp_config = model_template["hyperparameters"]

    def objective(trial):
        hyperparameters = {}
        for hp in hp_config:
            hp_value = hp_config[hp]
            if isinstance(hp_value, dict):
                assert hp_value["type"] in {"int", "float"}
                suggestion = trial.suggest_int(hp, *hp_value["range"]) if hp_value["type"] == "int" else trial.suggest_float(hp, *hp_value["range"])
                hyperparameters[hp] = 2**suggestion if hp_value.get("exponentiate", False) else suggestion
            else:
                hyperparameters[hp] = hp_value

        model = Model(n_timesteps=n_timesteps, **hyperparameters).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        best_val_loss = np.inf
        for epoch in range(nepochs):
            train(training_loader, model, loss_fn, optimizer, device)

            val_loss = evaluate(validation_loader, model, loss_fn, device)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=ntrials, catch=(RuntimeError,))

    best_params = {}
    for hp in hp_config:
        hp_value = hp_config[hp]
        if isinstance(hp_value, dict):
            best_params[hp] = 2**study.best_params[hp] if hp_value.get("exponentiate", False) else study.best_params[hp]
        else:
            best_params[hp] = hp_value

    return study.best_value, {"name" : model_template["name"] , "hyperparameters" : best_params}