import torch
import torch.nn as nn


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
        self.mask_non = (torch.tensor([name_to_idx[n] for n in self.non_monotonic_vars], dtype=torch.long)
                         if self.non_monotonic_vars else None)
        self.mask_pos = (torch.tensor([name_to_idx[n] for n in self.positive_monotonic_vars], dtype=torch.long)
                         if self.positive_monotonic_vars else None)
        self.mask_neg = (torch.tensor([name_to_idx[n] for n in self.negative_monotonic_vars], dtype=torch.long)
                         if self.negative_monotonic_vars else None)

        # Define hidden-layer branches (Linear layer only; tanh applied in forward)
        # Can add more layers if desired, but need to enforce constraints on all layers for monotonic branches
        self.lin_non = (nn.Linear(len(self.non_monotonic_vars), hidden_non)
                        if self.non_monotonic_vars and hidden_non > 0 else None)
        self.lin_pos = (nn.Linear(len(self.positive_monotonic_vars), hidden_pos)
                        if self.positive_monotonic_vars and hidden_pos > 0 else None)
        self.lin_neg = (nn.Linear(len(self.negative_monotonic_vars), hidden_neg)
                        if self.negative_monotonic_vars and hidden_neg > 0 else None)

        # Determine total hidden dimension
        total_h = ((self.lin_non.out_features if self.lin_non else 0) +
                   (self.lin_pos.out_features if self.lin_pos else 0) +
                   (self.lin_neg.out_features if self.lin_neg else 0))

        if total_h == 0:
            raise ValueError("At least one branch must have > 0 hidden units.")

        # Final output layer: maps concatenated hidden units → 1 logit
        # logit = w₁*h₁ + w₂*h₂ + ... + b
        self.out = nn.Linear(total_h, 1)

        # Initialize weights (Xavier uniform)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier initialization to all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steps:
            1. Takes the input and splits the features into monotonicity groups.
            2. Sends each group into its corresponding branch neural network.
            3. Applies a Linear layer + tanh on each branch.
            4. Concatenates all branch outputs.
            5. Sends them into a final linear layer to compute a logit.
            6. Applies sigmoid to return a probability.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, num_variables],
            columns in the same order as `self.all_variables`.

        Returns
        -------
        torch.Tensor
            Probability in [0, 1] of shape [batch_size, 1].
        """
        branches = []

        # Apply each branch if present
        if self.lin_non is not None:
            h_non = torch.tanh(self.lin_non(x.index_select(1, self.mask_non.to(x.device))))  # type: ignore
            branches.append(h_non)

        if self.lin_pos is not None:
            h_pos = torch.tanh(self.lin_pos(x.index_select(1, self.mask_pos.to(x.device))))  # type: ignore
            branches.append(h_pos)

        if self.lin_neg is not None:
            h_neg = torch.tanh(self.lin_neg(x.index_select(1, self.mask_neg.to(x.device))))  # type: ignore
            branches.append(h_neg)

        # Concatenate hidden representations (or use single branch)
        h = branches[0] if len(branches) == 1 else torch.cat(branches, dim=1)

        logits = self.out(h)
        probs = torch.sigmoid(logits)
        return probs

    @torch.no_grad()
    def monotonic_constraint(self) -> None:
        """
        Enforce monotonicity constraints by projecting parameters back into
        the feasible region. Should be called after each optimizer step.

        Constraints:
        ------------
        - Positive branch:
            * First-layer weights: >= 0
            * Output weights from this section: >= 0

        - Negative branch:
            * First-layer weights: <= 0
            * Output weights from this section: >= 0

        Biases are left unconstrained (bias does not affect derivative sign).
        """
        # Constrain first-layer weights
        if self.lin_pos is not None:
            self.lin_pos.weight.clamp_(min=0.0)

        if self.lin_neg is not None:
            self.lin_neg.weight.clamp_(max=0.0)

        # Constrain output-layer weights for monotonic sections
        start = 0

        # Skip non-monotonic part
        if self.lin_non is not None:
            start += self.lin_non.out_features

        # Positive monotonic section — enforce weights >= 0
        if self.lin_pos is not None:
            end = start + self.lin_pos.out_features
            w = self.out.weight[:, start:end]
            w.clamp_(min=0.0)
            self.out.weight[:, start:end] = w
            start = end

        # Negative monotonic section — enforce weights >= 0
        if self.lin_neg is not None:
            end = start + self.lin_neg.out_features
            w = self.out.weight[:, start:end]
            w.clamp_(min=0.0)
            self.out.weight[:, start:end] = w

    def fit(
        self,
        x_tr: torch.Tensor,
        y_tr: torch.Tensor,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 1e-3,
        optimizer_cls: torch.optim.Optimizer = torch.optim.RMSprop,  # or torch.optim.Adam
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
        self.train()

        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # ---- Optimizer
        optimizer = optimizer_cls(self.parameters(), lr=lr)

        # ---- Initial projection (recommended with projection approach)
        # If you switch to reparameterization, REMOVE this call.
        self.monotonic_constraint()

        history: dict[str, list[float]] = {"epoch_loss": []}

        for epoch in range(epochs):
            running_loss, n_batches = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float().view(-1, 1)

                optimizer.zero_grad()

                p = self(xb)  # probabilities [B,1] (your forward applies sigmoid)

                loss = torch.mean((p - yb)**2)               # simple MSE; replace with weighted MSE if needed
                # weighted_mse_loss = torch.div(torch.sum(rebalance_weights*((p - yb) ** 2)), torch.sum(rebalance_weights))

                loss.backward()
                optimizer.step()

                # ---- Projection step (only for clamp-based constraints)
                # If you switch to reparameterization, REMOVE this call.
                self.monotonic_constraint()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            history["epoch_loss"].append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")

        return history

    # @torch.no_grad()
    # def predict_proba(self, x):
    #     """Return probabilities for a torch tensor or numpy array."""
    #     self.eval()
    #     if isinstance(x, np.ndarray):
    #         x = torch.from_numpy(x).float()
    #     x = x.to(next(self.parameters()).device)
    #     p = self(x)
    #     self.train()
    #     return p.detach().cpu().numpy()