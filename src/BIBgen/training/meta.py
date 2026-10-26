from __future__ import annotations

from typing import Any, Callable
import json
import logging

import torch
import numpy as np

from BIBgen import models as model_classes
from BIBgen.training.dataloaders import BaseDataLoader
from BIBgen.training.core import train, evaluate

logger = logging.getLogger(__name__)

def load_empty_model(model_config_path : str, n_timesteps : int | None = None, mu : torch.Tensor | None = None, std : torch.Tensor | None = None):
    assert model_config_path.endswith(".json")
    with open(model_config_path, "r") as fin:
        model_config = json.load(fin)

    model_cls = getattr(model_classes, model_config["name"])
    model_args = model_config["hyperparameters"]
    if model_cls.need_n_timesteps():
        assert n_timesteps is not None
        model_args["n_timesteps"] = n_timesteps
    if model_cls.need_mu_std():
        assert mu is not None and std is not None
        model_args["std"] = std
        model_args["mu"] = mu

    return model_cls(**model_args)

def hyperoptimize(
    model_template : dict[str, Any],
    training_loader : BaseDataLoader,
    validation_loader : BaseDataLoader,
    loss_fn : Callable[..., torch.Tensor],
    device : torch.device,
    nepochs : int = 20,
    ntrials : int = 40,
    storage : str | None = None,
    study_name : str | None = None,
    timeout : float | None = None,
    n_timesteps : int | None = None,
) -> tuple[float | None, dict[str, Any] | None, int]:
    """
    Searches the hyperparameter ranges of a model template with optuna, minimizing validation loss.

    With ``storage``, every finished trial is saved there, and a study that already exists in it is
    resumed rather than restarted, so one search can be spread over several jobs. ``ntrials`` is
    then the total over all of them, not the number run by this call.

    Parameters
    ----------
    model_template : dict
        ``{"name": ..., "hyperparameters": {...}}``. A hyperparameter given as a dict
        ``{"type": "int" or "float", "range": [low, high], "exponentiate": bool}`` is searched,
        as ``2**value`` if ``exponentiate`` is true. Any other value is passed to the model as is.
    training_loader : BaseDataLoader
        Loader for training data
    validation_loader : BaseDataLoader
        Loader for validation data
    loss_fn : Callable
        Loss function called on model output as ``loss_fn(pred, X, y, tau, x_0)``
    device : torch.device
        Device on which to train
    nepochs : int
        Number of epochs to train each trial for
    ntrials : int
        Total number of trials in the study. Completed, pruned and errored trials count towards
        it. A trial cut off because its process was killed does not count, and is run again with
        the same hyperparameters when the study is resumed.
    storage : str or None
        Optuna storage URL, e.g. ``"sqlite:///study.db"``. None keeps the study in memory only.
    study_name : str or None
        Name of the study within ``storage``
    timeout : float or None
        Seconds after which no new trial is started. A trial already running is allowed to finish.
    n_timesteps : int, optional
        Number of diffusion timesteps, passed to the model

    Returns
    -------
    best_val_loss : float or None
        Best validation loss over completed trials, or None if no trial has completed yet
    best_config : dict or None
        Model config of the best trial, in the format read by ``load_empty_model``, or None if no
        trial has completed yet
    ntrials_done : int
        Number of trials counted towards ``ntrials`` so far. The study is finished once this
        reaches ``ntrials``.
    """
    import optuna
    from optuna.trial import TrialState

    Model = getattr(model_classes, model_template["name"])
    hp_config = model_template["hyperparameters"]

    def to_model_value(hp : str, value : Any) -> Any:
        return 2**value if hp_config[hp].get("exponentiate", False) else value

    def objective(trial : optuna.Trial) -> float:
        hyperparameters = {}
        for hp in hp_config:
            hp_value = hp_config[hp]
            if isinstance(hp_value, dict):
                assert hp_value["type"] in {"int", "float"}
                suggestion = trial.suggest_int(hp, *hp_value["range"]) if hp_value["type"] == "int" else trial.suggest_float(hp, *hp_value["range"])
                hyperparameters[hp] = to_model_value(hp, suggestion)
            else:
                hyperparameters[hp] = hp_value

        model_params = dict(hyperparameters)
        if Model.need_mu_std():
            model_params["mu"] = training_loader.mu()
            model_params["std"] = training_loader.std()
        if Model.need_n_timesteps():
            assert n_timesteps is not None
            model_params["n_timesteps"] = n_timesteps

        try:
            model = Model(**model_params).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

            best_val_loss = np.inf
            for epoch in range(nepochs):
                train(training_loader, model, loss_fn, optimizer, device)

                val_loss = evaluate(validation_loader, model, loss_fn, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                trial.report(best_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except RuntimeError as err:
            # Marks the failure as an error of the trial itself (e.g. out of memory, nonfinite loss),
            # which counts towards ntrials, unlike a trial cut off by its process being killed.
            trial.set_user_attr("error", repr(err))
            raise

        return float(best_val_loss)

    def count_done(study : optuna.Study) -> int:
        finished = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
        return sum(t.state != TrialState.FAIL or "error" in t.user_attrs for t in finished)

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    # Only one process works on a study at a time, so a trial still marked as running was cut off
    # when an earlier process was killed. Fail it and queue its hyperparameters to be tried again.
    for interrupted in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(interrupted.number, state=TrialState.FAIL)
        study.enqueue_trial(interrupted.params, user_attrs={"retry_of": interrupted.number})
        logger.warning("Trial %d was interrupted; queued %s to run again", interrupted.number, interrupted.params)

    nremaining = ntrials - count_done(study)
    if nremaining > 0:
        study.optimize(objective, n_trials=nremaining, timeout=timeout, catch=(RuntimeError,), gc_after_trial=True)

    if not any(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False)):
        return None, None, count_done(study)

    best_params = {}
    for hp in hp_config:
        hp_value = hp_config[hp]
        if isinstance(hp_value, dict):
            best_params[hp] = to_model_value(hp, study.best_params[hp])
        else:
            best_params[hp] = hp_value

    return study.best_value, {"name" : model_template["name"] , "hyperparameters" : best_params}, count_done(study)
