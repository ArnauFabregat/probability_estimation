import copy

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.schemas import OptimizerParams


class MonotonicLinear(nn.Module):
    """
    A linear layer with monotonicity constraints enforced through
    softplus-based weight reparameterization.

    This layer ensures that the sign of the partial derivative of the output
    with respect to each input dimension remains fixed, making it suitable
    for monotonic neural networks.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    sign : str, optional (default="+")
        Monotonicity direction:
            "+" → enforces ∂y/∂x >= 0 (positive monotonicity)
            "-" → enforces ∂y/∂x <= 0 (negative monotonicity)

    Forward Input
    -------------
    x : torch.Tensor
        Input tensor of shape [batch_size, in_features].

    Returns
    -------
    torch.Tensor
        Output tensor of shape [batch_size, out_features].

    Notes
    -----
    - `softplus` guarantees smooth, strictly positive transformed weights.
    - Bias parameters remain unconstrained (do not affect monotonicity).
    - Positive monotonicity:
          weight =  softplus(raw_weight)
    - Negative monotonicity:
          weight = -softplus(raw_weight)
    """

    def __init__(self, in_features: int, out_features: int, sign: str = "+") -> None:
        super().__init__()

        if sign not in {"+", "-"}:
            raise ValueError("sign must be '+' or '-'.")

        self.in_features = in_features
        self.out_features = out_features
        self.sign = sign

        # Unconstrained raw weight; transformed via softplus in forward()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Bias is unconstrained
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply monotonic linear transformation to input.

        Parameters
        ----------
        x : torch.Tensor
            Shape [batch_size, in_features]

        Returns
        -------
        torch.Tensor:
            Shape [batch_size, out_features].
        """

        # softplus ensures strictly positive weights
        weight = F.softplus(self.raw_weight)

        # enforce monotonic direction
        if self.sign == "-":
            weight = -weight

        # Linear transformation: y = xW^T + b
        return x @ weight.T + self.bias


class MonotonicNN(nn.Module):
    """
    Monotonic neural network with three optional branches enforcing different
    monotonicity behaviors over subsets of input variables.

    Branches
    --------
    1. Non‑monotonic branch:
         - Inputs: variables in `non_monotonic_vars`
         - No constraints on weights
         - Architecture: Linear → tanh

    2. Positive‑monotonic branch:
         - Inputs: variables in `positive_monotonic_vars`
         - First-layer weights >= 0
         - Output-layer weights >= 0
         - Ensures ∂output/∂variable >= 0

    3. Negative‑monotonic branch:
         - Inputs: variables in `negative_monotonic_vars`
         - First-layer weights <= 0
         - Output-layer weights >= 0
         - Ensures ∂output/∂variable <= 0

    Output
    ------
    Produces a single logit, then applies a sigmoid to obtain a probability
    in the range [0, 1].

    Notes
    -----
    - Branch widths are configurable.
    - Feature names must match the column order in the input tensor `x`.
    - Monotonicity is enforced at the layer level via `MonotonicLinear`.
    """

    def __init__(
        self,
        all_variables: list[str],
        non_monotonic_vars: list[str] = [],
        positive_monotonic_vars: list[str] = [],
        negative_monotonic_vars: list[str] = [],
        hidden_non: int = 16,
        hidden_pos: int = 8,
        hidden_neg: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        all_variables : list[str]
            Full ordered list of feature names matching columns of the input tensor.

        non_monotonic_vars : list[str]
            Variables without monotonicity constraints.

        positive_monotonic_vars : list[str]
            Variables required to have non-negative effect on the output.

        negative_monotonic_vars : list[str]
            Variables required to have non-positive effect on the output.

        hidden_non, hidden_pos, hidden_neg : int
            Number of hidden units in the non‑monotonic, positive‑monotonic,
            and negative‑monotonic branches respectively.
        """
        super().__init__()

        # Store variable lists
        self.all_variables = all_variables
        self.non_monotonic_vars = non_monotonic_vars or []
        self.positive_monotonic_vars = positive_monotonic_vars or []
        self.negative_monotonic_vars = negative_monotonic_vars or []

        # Map variable names to their column indices
        name_to_idx = {name: i for i, name in enumerate(self.all_variables)}

        # Masks selecting the input columns for each branch
        if self.non_monotonic_vars:
            self.register_buffer(
                "mask_non", torch.tensor([name_to_idx[n] for n in self.non_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_non = None

        if self.positive_monotonic_vars:
            self.register_buffer(
                "mask_pos", torch.tensor([name_to_idx[n] for n in self.positive_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_pos = None

        if self.negative_monotonic_vars:
            self.register_buffer(
                "mask_neg", torch.tensor([name_to_idx[n] for n in self.negative_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_neg = None

        # Define hidden-layer branches (Linear layer only; tanh applied in forward)
        # Can add more layers if desired, but need to enforce constraints on all layers for monotonic branches
        self.lin_non = (
            nn.Linear(len(self.non_monotonic_vars), hidden_non) if self.non_monotonic_vars and hidden_non > 0 else None
        )
        self.lin_pos = (
            MonotonicLinear(
                in_features=len(positive_monotonic_vars),
                out_features=hidden_pos,
                sign="+",
            )
            if self.positive_monotonic_vars and hidden_pos > 0
            else None
        )
        self.lin_neg = (
            MonotonicLinear(
                in_features=len(negative_monotonic_vars),
                out_features=hidden_neg,
                sign="-",
            )
            if self.negative_monotonic_vars and hidden_neg > 0
            else None
        )

        # Output layers – SUM of three logits
        self.out_non = nn.Linear(hidden_non, 1) if self.lin_non else None
        self.out_pos = MonotonicLinear(hidden_pos, 1, sign="+") if self.lin_pos else None
        self.out_neg = MonotonicLinear(hidden_neg, 1, sign="+") if self.lin_neg else None

        # Initialize weights (Xavier uniform)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier uniform initialization for all layers.

        - nn.Linear layers: initialize `.weight` and `.bias`
        - MonotonicLinear layers: initialize `.raw_weight` and `.bias`
        """
        for m in self.modules():
            # Standard Linear layers
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # Custom monotonic layers
            if isinstance(m, MonotonicLinear):
                nn.init.xavier_uniform_(m.raw_weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the monotonic neural network.

        Steps
        -----
        1. Split the input tensor into feature subsets based on their
        monotonicity group.

        2. Pass each subset through its corresponding branch:
            - Non‑monotonic:      Linear → tanh
            - Positive‑monotonic: MonotonicLinear(sign="+") → tanh
            - Negative‑monotonic: MonotonicLinear(sign="-") → tanh

        3. Send branch hidden outputs to their respective output layers.

        4. Sum logits from all active branches (not concatenated), preserving
        monotonicity.

        5. Return the summed logit (sigmoid is applied externally if needed).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_features].

        Returns
        -------
        torch.Tensor
            Logit tensor of shape [batch_size, 1].
        """

        # Apply each branch if present
        total_logit = torch.zeros((x.shape[0], 1), device=x.device)

        if self.lin_non:
            h = torch.tanh(self.lin_non(x[:, self.mask_non]))
            total_logit += self.out_non(h)  # type: ignore

        if self.lin_pos:
            h = torch.tanh(self.lin_pos(x[:, self.mask_pos]))
            total_logit += self.out_pos(h)  # type: ignore

        if self.lin_neg:
            h = torch.tanh(self.lin_neg(x[:, self.mask_neg]))
            total_logit += self.out_neg(h)  # type: ignore

        return total_logit

    def fit(
        self,
        x_tr: torch.Tensor,
        y_tr: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        pos_weight: float = 1.0,
        epochs: int = 5,
        optimizer_params: OptimizerParams = OptimizerParams(),
        shuffle: bool = True,
        num_workers: int = 0,
        device: str | torch.device = "cpu",  # "cuda" if available
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the model using minibatch gradient descent with optional validation
        and early stopping.

        Parameters
        ----------
        x_tr : torch.Tensor
            Training input tensor of shape [N, d].

        y_tr : torch.Tensor
            Training target tensor of shape [N].

        x_val : torch.Tensor, optional
            Validation input tensor.

        y_val : torch.Tensor, optional
            Validation target tensor.

        pos_weight : float, default=1.0
            Positive class weight for BCEWithLogitsLoss.

        epochs : int
            Number of epochs to train.

        optimizer_params : OptimizerParams
            Hyperparameters for the optimizer and early stopping.

        shuffle : bool, default=True
            Whether to shuffle minibatches.

        num_workers : int, default=0
            DataLoader worker count.

        device : str or torch.device, default="cpu"
            Training device.

        verbose : bool, default=True
            Print progress and loss values during training.

        Returns
        -------
        dict[str, list[float]]
            Training history containing:
                - "train_loss" : list of training losses per epoch
                - "val_loss"   : list of validation losses per epoch (if provided)
        """
        self.to(device)

        # Ensure reproducibility
        torch.manual_seed(42)
        numpy.random.seed(42)

        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=optimizer_params.batch_size, shuffle=shuffle, num_workers=num_workers
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=optimizer_params.lr, weight_decay=optimizer_params.weight_decay
        )

        # Initialize history dictionary
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Early Stopping Variables
        best_val_loss: float = float("inf")
        epochs_no_improve: int = 0
        best_model_state = None

        # Use BCEWithLogitsLoss which combines a sigmoid layer and the BCELoss in one single class.
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        for epoch in range(epochs):
            # --- Training Phase ---
            self.train()
            running_loss: float = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float().view(-1, 1)
                optimizer.zero_grad()

                logits = self.forward(xb)
                loss = criterion(logits, yb)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(loader)
            history["train_loss"].append(avg_train_loss)

            # --- Validation Phase ---
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    x_v = x_val.to(device).float()
                    y_v = y_val.to(device).float().view(-1, 1)
                    logits_val = self.forward(x_v)
                    val_loss = criterion(logits_val, y_v)
                    history["val_loss"].append(val_loss.item())

                # Check Early Stopping
                if val_loss < (best_val_loss - optimizer_params.min_delta):
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Deep copy the state so we can restore the best version later
                    best_model_state = copy.deepcopy(self.state_dict())
                else:
                    epochs_no_improve += 1

                if verbose:
                    print(f"Epoch {epoch + 1} | Train: {avg_train_loss:.5f} | Val: {val_loss.item():.5f}")

                if epochs_no_improve >= optimizer_params.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                    # Restore the best weights before exiting
                    self.load_state_dict(best_model_state)  # type: ignore
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1} | Train loss: {avg_train_loss:.5f}")

        return history

    @torch.no_grad()
    def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Evaluate the model and return predicted probabilities.

        Parameters
        ----------
        x : numpy.ndarray
            Input array of shape [N, d] containing N samples and d features.

        Returns
        -------
        numpy.ndarray
            Probability of belonging to the positive class for each sample.
        """
        self.eval()
        device = next(self.parameters()).device

        x_input = torch.from_numpy(x).float().to(device)
        logits = self.forward(x_input)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    @torch.no_grad()
    def predict_logits(self, x: numpy.ndarray) -> torch.Tensor:
        """
        Return tensor of raw logits required for temperature scaling.

        Parameters
        ----------
        x : numpy.ndarray
            Input array of shape [N, d] containing N samples and d features.

        Returns
        -------
        torch.Tensor
            Raw logits belonging to the positive class for each sample.
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.from_numpy(x).float().to(device)
        logits = self.forward(x_t)
        return logits
