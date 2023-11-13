import pathlib

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self,
            patience=7,
            delta=0.0,
            metrics=["loss", "accuracy"],
            es_metric="loss",
            verbose=False,
            path="~/Documents/temp",
            filename="checkpoint.pt",
            trace_func=print,
            bool_save_checkpoint=False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.delta = delta
        self.metrics = metrics
        self.es_metric = es_metric
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = True
        self.val_loss_min = np.Inf
        self.path = pathlib.Path(path)
        self.filename = filename
        self.trace_func = trace_func

        # initialize internal list of models
        self.bool_save_checkpoint = bool_save_checkpoint
        self.best_model_weights = None

    def __call__(
            self,
            val_metric,
            model,
            mode="min",
    ):
        if mode == "min":
            score = -val_metric
        elif mode == "max":
            score = val_metric
        else:
            raise ValueError("mode must be min or max")

        # initialize
        if self.best_score is None:
            self.best_score = score
            self.save_model(val_metric, model)

        # terminate if no improvement
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True

        # if model has improved
        else:
            self.best_score = score
            self.save_model(val_metric, model)
            self.counter = 0

    def save_model(
            self,
            val_loss,
            model,
    ):
        """
        Saves model when validation loss decrease.
        """

        if self.bool_save_checkpoint:
            self._save_checkpoint(val_loss, model)
        else:
            self._save_model_to_self(val_loss, model)

    def _save_checkpoint(
            self,
            val_loss,
            model,
    ):
        # create output directory if it doesn't exist
        self.path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), str(self.path / self.filename))
        self.val_loss_min = val_loss

    def _save_model_to_self(
            self,
            val_loss,
            model,
    ):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        # append to internal list
        self.val_loss_min = val_loss
        self.best_model_weights = model.state_dict()

    def load_model(
            self,
            model,
    ):
        """
        Loads model when validation loss decrease.
        """

        if self.bool_save_checkpoint:
            model = self._load_checkpoint(model)
        else:
            model = self._load_model_from_self(model)

        return model

    def _load_checkpoint(
            self,
            model,
    ):
        if self.verbose:
            self.trace_func(f"Loading model from {str(self.path / self.filename)} ...")
        model.load_state_dict(torch.load(str(self.path / self.filename)))

        return model

    def _load_model_from_self(
            self,
            model,
    ):
        if self.verbose:
            self.trace_func(f"Loading model from self ...")

        model.load_state_dict(self.best_model_weights)

        return model
