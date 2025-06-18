import torch.nn as nn

# Defining Early Stopping Class
class EarlyStopping(nn.Module):

  def __init__(self, patience=4, delta=0.001):
    self.patience = patience # Number of steps to wait after last improvement before stopping the training
    self.delta = delta # Minimum change required to considered as improvements
    self.best_loss = float('inf') # Best Validation Loss seen so far
    self.counter = 0 # Number of Steps without Improvements
    self.early_stop = False # If True, it will stop the training

  def __call__(self, val_loss):
    if val_loss < self.best_loss - self.delta:
      self.best_loss = val_loss
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True