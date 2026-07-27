"""Early stopping with checkpointing, shared by MDP pretraining and
segmentation fine-tuning (patience 20 in the paper)."""

import numpy as np
import torch


class EarlyStopping:
    """Stop training when the validation loss has not decreased for
    ``patience`` consecutive epochs; keep the best checkpoint on disk."""

    def __init__(self, patience=20, verbose=False, delta=0.0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} '
                  f'--> {val_loss:.6f}). Saving model ...')
        if epoch is not None:
            weight_path = f"{self.path[:-4]}_{epoch}_{val_loss:.5f}.pth"
        else:
            weight_path = self.path
        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'model_state_dict': model.state_dict(),
        }, weight_path)
        self.val_loss_min = val_loss
