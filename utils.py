from typing import Optional
import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat


class OptimizerParams(BaseModel):
    lr: PositiveFloat = 1e-3
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: PositiveInt = 1024
    patience: PositiveInt = 10
    min_delta: float = 1e-4  # Minimum change to qualify as improvement


class MonotonicLinear(nn.Module):
    """
    A Linear layer with **built-in monotonicity constraints** applied through
    softplus-based weight reparameterization.

    This layer guarantees that the sign of the partial derivative with respect to
    each input dimension remains fixed, which is essential for monotonic neural
    networks.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    sign : str, optional (default="+")
        Monotonicity direction:
            "+" enforces positive monotonicity (∂y/∂x >= 0)
            "-" enforces negative monotonicity (∂y/∂x <= 0)

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
    - softplus ensures smoothness and better gradient behavior than exp().
    - bias is unconstrained because it does not affect monotonicity.
    - For **positive monotonicity** w.r.t. the inputs:
        MonotonicLinear(..., sign="+")
        weight =  softplus(raw_weight)
    - For **negative monotonicity** w.r.t. the inputs:
        MonotonicLinear(..., sign="-")
        weight = -softplus(raw_weight)
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 sign: str = "+") -> None:
        super().__init__()

        if sign not in {"+", "-"}:
            raise ValueError("sign must be '+' or '-'.")

        self.in_features = in_features
        self.out_features = out_features
        self.sign = sign

        # Unconstrained raw weight; transformed via softplus in forward()
        self.raw_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

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
    Monotonic Neural Network with three optional branches enforcing different
    monotonicity behaviors with respect to subsets of input variables.

    Branches:
    ---------
    1. Non‑monotonic branch:
         - Inputs: variables in `non_monotonic_vars`
         - No constraints on weights (fully flexible)
         - One hidden layer (Linear + tanh)

    2. Positive‑monotonic branch:
         - Inputs: variables in `positive_monotonic_vars`
         - First‑layer weights constrained to be >= 0
         - Output‑layer weights from this branch constrained to be >= 0
         - Ensures ∂output/∂variable >= 0

    3. Negative‑monotonic branch:
         - Inputs: variables in `negative_monotonic_vars`
         - First‑layer weights constrained to be <= 0
         - Output‑layer weights from this branch constrained to be >= 0
         - Ensures ∂output/∂variable <= 0

    Output:
    -------
    - A probability in [0, 1] obtained by applying sigmoid to the final logit.

    Notes:
    ------
    - Branch sizes are configurable.
    - Variables must appear in `all_variables` in the same order as columns of x.
    - Monotonicity constraints are enforced **after each optimizer step** via
      `monotonic_constraint()`.
    """

    def __init__(self,
                 all_variables: list[str],
                 non_monotonic_vars: list[str] = [],
                 positive_monotonic_vars: list[str] = [],
                 negative_monotonic_vars: list[str] = [],
                 hidden_non: int = 16,
                 hidden_pos: int = 8,
                 hidden_neg: int = 8) -> None:
        """
        Parameters
        ----------
        all_variables : list[str]
            Full list of variable names, matching the column order of the input tensor.

        non_monotonic_vars : list[str]
            Variables with no monotonicity constraints.

        positive_monotonic_vars : list[str]
            Variables required to have a non‑negative effect on the output.

        negative_monotonic_vars : list[str]
            Variables required to have a non‑positive effect on the output.

        hidden_non, hidden_pos, hidden_neg : int
            Number of hidden units per branch.
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
                "mask_non",
                torch.tensor([name_to_idx[n] for n in self.non_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_non = None

        if self.positive_monotonic_vars:
            self.register_buffer(
                "mask_pos",
                torch.tensor([name_to_idx[n] for n in self.positive_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_pos = None

        if self.negative_monotonic_vars:
            self.register_buffer(
                "mask_neg",
                torch.tensor([name_to_idx[n] for n in self.negative_monotonic_vars], dtype=torch.long)
            )
        else:
            self.mask_neg = None

        # Define hidden-layer branches (Linear layer only; tanh applied in forward)
        # Can add more layers if desired, but need to enforce constraints on all layers for monotonic branches
        self.lin_non = (
            nn.Linear(
                len(self.non_monotonic_vars),
                hidden_non
            )
            if self.non_monotonic_vars and hidden_non > 0 else None
        )
        self.lin_pos = (
            MonotonicLinear(
                in_features=len(positive_monotonic_vars),
                out_features=hidden_pos,
                sign="+",
            )
            if self.positive_monotonic_vars and hidden_pos > 0 else None
        )
        self.lin_neg = (
            MonotonicLinear(
                in_features=len(negative_monotonic_vars),
                out_features=hidden_neg,
                sign="-",
            )
            if self.negative_monotonic_vars and hidden_neg > 0 else None
        )

        # Output layers – SUM of three logits
        self.out_non = nn.Linear(hidden_non, 1) if self.lin_non else None
        self.out_pos = MonotonicLinear(hidden_pos, 1, sign="+") if self.lin_pos else None
        self.out_neg = MonotonicLinear(hidden_neg, 1, sign="+") if self.lin_neg else None

        # Initialize weights (Xavier uniform)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Xavier initialization to all Linear layers, including
        custom MonotonicLinear layers by initializing their raw weights.

        - nn.Linear → initialize .weight and .bias
        - MonotonicLinear → initialize .raw_weight and .bias
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
        Compute the forward pass of the monotonic neural network.

        Steps:
        ------
        1. Split the input tensor into subsets of features according to their
        monotonicity groups:
            - non‑monotonic variables
            - positively monotonic variables
            - negatively monotonic variables

        2. Each subset is passed through its corresponding branch:
            - Non‑monotonic branch:      Linear → tanh
            - Positive‑monotonic branch: MonotonicLinear(sign="+") → tanh
            - Negative‑monotonic branch: MonotonicLinear(sign="-") → tanh

        3. Each branch produces a hidden representation, which is fed into its
        own output layer:
            - out_non: standard Linear
            - out_pos: MonotonicLinear(sign="+")
            - out_neg: MonotonicLinear(sign="+")
        (Positive sign is correct for both monotonic branches because the
        hidden-layer sign determines the derivative direction.)

        4. The three output logits are **summed**, not concatenated, preserving
        monotonicity and avoiding interaction terms that could break it.

        5. A final sigmoid is applied to convert the summed logit into a
        probability in [0, 1].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_variables], where the columns
            appear in the same order as `self.all_variables`.

        Returns
        -------
        torch.Tensor
            Tensor of shape [batch_size, 1] containing the predicted probability
            for each input sample.
        """

        # Apply each branch if present
        total_logit: float = 0.0

        if self.lin_non:
            h = torch.tanh(self.lin_non(x.index_select(1, self.mask_non)))  # type: ignore
            total_logit += self.out_non(h)  # type: ignore

        if self.lin_pos:
            h = torch.tanh(self.lin_pos(x.index_select(1, self.mask_pos)))  # type: ignore
            total_logit += self.out_pos(h)  # type: ignore

        if self.lin_neg:
            h = torch.tanh(self.lin_neg(x.index_select(1, self.mask_neg)))  # type: ignore
            total_logit += self.out_neg(h)  # type: ignore

        return torch.sigmoid(total_logit)  # type: ignore

    def fit(
        self,
        x_tr: torch.Tensor,
        y_tr: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 5,
        optimizer_params: OptimizerParams = OptimizerParams(),
        shuffle: bool = True,
        num_workers: int = 0,
        device: str | torch.device = "cpu",   # "cuda" if available
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the model using a simple minibatch loop with monotonic projection after each step.

        Parameters
        ----------
        epochs : int
            Number of epochs.
        batch_size : int
            Minibatch size.
        lr : float
            Learning rate.
        optimizer_cls : torch.optim.Optimizer
            Optimizer class to instantiate (e.g., RMSprop, Adam).
        device : str | torch.device
            Device to train on.
        verbose : bool
            Print epoch loss.

        Returns
        -------
        dict
            Training history with avg loss per epoch.
        """
        self.to(device)

        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=optimizer_params.batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=optimizer_params.lr,
            weight_decay=optimizer_params.weight_decay
        )

        # Initialize history dictionary
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Early Stopping Variables
        best_val_loss: float = float('inf')
        epochs_no_improve: int = 0
        best_model_state = None

        for epoch in range(epochs):
            # --- Training Phase ---
            self.train()
            running_loss: float = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float().view(-1, 1)
                optimizer.zero_grad()

                # probabilities [B,1] (applies sigmoid)
                p = self.forward(xb)
                loss = torch.mean((p - yb)**2)               # simple MSE; replace with weighted MSE if needed
                # rebalance_weights <- compute_weights for yb based on model_rebalance
                # weighted_mse_loss = torch.div(torch.sum(rebalance_weights*((p-yb)**2)), torch.sum(rebalance_weights))

                loss.backward()  # type: ignore
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
                    p_val = self.forward(x_v)
                    avg_val_loss = torch.mean((p_val - y_v)**2).item()
                    history["val_loss"].append(avg_val_loss)

                # Check Early Stopping
                if avg_val_loss < (best_val_loss - optimizer_params.min_delta):
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Deep copy the state so we can restore the best version later
                    best_model_state = copy.deepcopy(self.state_dict())
                else:
                    epochs_no_improve += 1

                if verbose:
                    print(f"Epoch {epoch+1} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

                if epochs_no_improve >= optimizer_params.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    # Restore the best weights before exiting
                    self.load_state_dict(best_model_state)  # type: ignore
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1} | Train loss: {avg_train_loss:.5f}")

        return history

    @torch.no_grad()
    def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Eval the model and transform its output to obtain the statistical probability of belonging to the minority
        class for each sample.

        Parameters
        ----------
        x : numpy.ndarray)
            The input matrix NxM containing M dimensions for N sampels.

        Returns
        -------
        numpy.ndarray
            The probability of belonging to the minority class for each sample evaluated.
        """
        self.eval()
        device = next(self.parameters()).device

        x_input = torch.from_numpy(x).float().to(device)
        output = self.forward(x_input).cpu().numpy()

        return output
