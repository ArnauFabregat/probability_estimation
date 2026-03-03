import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray


class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for calibrating neural‑network logits.

    This method takes **raw logits** (pre‑sigmoid scores) and applies a learned
    scalar temperature `T > 0` such that:

        calibrated_logits = logits / T
        calibrated_probs = sigmoid(calibrated_logits)

    Temperature scaling is extremely low‑variance and works best when the model
    already has good ranking performance but is overconfident. It does NOT alter
    the order of predictions (monotonic mapping).

    Parameters
    ----------
    init_temp : float, default=1.0
        Initial value for the temperature parameter. Internally stored as
        `log_T` so that `T = softplus(log_T)` is guaranteed positive.

    Attributes
    ----------
    log_T : nn.Parameter
        Learnable parameter representing `log(temperature)` in unconstrained
        space.

    temperature : torch.Tensor (property)
        Current positive temperature value `T = softplus(log_T) + 1e-6`.

    Notes
    -----
    - Inputs may be either NumPy arrays or PyTorch tensors.
    - Inputs must be **logits**, not probabilities.
    - Outputs are always returned as **NumPy float64 arrays**.
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_temp)), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """
        Compute the temperature in constrained space.

        Returns
        -------
        T : torch.Tensor
            Positive scalar temperature value.
        """
        return torch.nn.functional.softplus(self.log_T) + 1e-6

    def fit(
        self,
        logits: torch.Tensor | NDArray[np.float64] | NDArray[np.float32],
        y: torch.Tensor | NDArray[np.int_] | NDArray[np.float32],
        max_iter: int = 50,
    ) -> None:
        """
        Fit the temperature parameter using binary cross‑entropy.

        Parameters
        ----------
        logits : array-like or Tensor, shape (n_samples,) or (n_samples, 1)
            Raw logits from a neural network. NOT probabilities.
        y : array-like or Tensor, shape (n_samples,)
            Ground truth binary labels (0/1).
        max_iter : int, default=50
            Maximum LBFGS optimization steps.

        Raises
        ------
        ValueError
            If shapes are inconsistent.
        """
        # Convert to tensors
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        logits = logits.float()
        y = y.float()

        # Normalize shapes
        if logits.dim() == 2:
            logits = logits.squeeze(1)
        y = y.view(-1)

        if logits.shape[0] != y.shape[0]:
            raise ValueError(f"logits and y must have same length. Got {logits.shape[0]} vs {y.shape[0]}.")

        # Detach logits from computation graph
        logits = logits.detach()

        optimizer = optim.LBFGS([self.log_T], lr=0.01, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nn.functional.binary_cross_entropy_with_logits(scaled, y)
            loss.backward()  # type: ignore
            return loss

        optimizer.step(closure)  # type: ignore

    @torch.no_grad()
    def predict_proba(
        self,
        logits: torch.Tensor | NDArray[np.float64] | NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """
        Predict calibrated probabilities from logits.

        Parameters
        ----------
        logits : array-like or Tensor, shape (n_samples,) or (n_samples, 1)
            Raw logits to calibrate.

        Returns
        -------
        probs : ndarray of shape (n_samples,)
            Calibrated probabilities in `[0, 1]`.

        Notes
        -----
        - Returned as `np.float64`.
        - Temperature scaling preserves the relative ordering of logits.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        logits = logits.float()

        if logits.dim() == 2:
            logits = logits.squeeze(1)

        scaled = logits / self.temperature
        probs = torch.sigmoid(scaled)

        return probs.cpu().numpy().reshape(-1).astype(float)
