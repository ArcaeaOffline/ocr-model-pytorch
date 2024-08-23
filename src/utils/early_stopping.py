import copy

import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(
        self,
        patience=15,
        verbose=False,
        delta=0,
        path=None,
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
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
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_state_dict = None

    def log(self, msg: str, *, force: bool = False):
        if force or self.verbose:
            self.trace_func(f"[EarlyStopping] {msg}")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return

        self.counter += 1
        self.log(f"Test loss didn't decrease ({self.counter}/{self.patience})")
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        self.log(
            f"Test loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Copying state dict..."
        )
        self.best_state_dict = copy.deepcopy(model.state_dict())

        if self.path is not None:
            self.log(f"Saving checkpoint to {self.path}...")
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def restore_model(self, model):
        return model.load_state_dict(self.best_state_dict)
