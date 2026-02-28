import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Logits-only temperature scaling.
    You pass logits, NOT features.
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        # log(T) to ensure T stays positive
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_temp)), dtype=torch.float32))

    @property
    def temperature(self):
        return torch.nn.functional.softplus(self.log_T) + 1e-6

    def fit(self, logits, y, max_iter=50):
        """
        logits: [N] or [N,1] (numpy or tensor)
        y:      [N]          (numpy or tensor)
        """
        # Convert to 1D tensors
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        logits = logits.float()
        y = y.float()

        if logits.dim() == 2:
            logits = logits.squeeze(1)
        y = y.view(-1)

        # detach logits from computation graph
        logits = logits.detach()

        optimizer = optim.LBFGS([self.log_T], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nn.functional.binary_cross_entropy_with_logits(scaled, y)
            loss.backward()
            return loss

        optimizer.step(closure)

    @torch.no_grad()
    def predict_proba(self, logits):
        """
        logits: [N] or [N,1] (numpy or tensor)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        logits = logits.float()

        if logits.dim() == 2:
            logits = logits.squeeze(1)

        scaled = logits / self.temperature
        probs = torch.sigmoid(scaled)
        return probs.cpu().numpy().reshape(-1)
