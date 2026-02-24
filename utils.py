import torch
import torch.nn as nn


class MonotonicNN(nn.Module):
    """
    Monotonic NN (one hidden layer per branch) with:
      - non-monotonic branch: unconstrained weights
      - positive-monotonic branch: first-layer weights >= 0, output weights from this branch >= 0
      - negative-monotonic branch: first-layer weights <= 0, output weights from this branch >= 0

    Forward returns probabilities in [0, 1] (sigmoid applied).
    """
    def __init__(self,
                 all_variables: list[str],
                 non_monotonic_vars: list[str] = [],
                 positive_monotonic_vars: list[str] = [],
                 negative_monotonic_vars: list[str] = [],
                 hidden_non: int = 16,
                 hidden_pos: int = 8,
                 hidden_neg: int = 8) -> None:
        super().__init__()
        # ---- store names
        self.all_variables = all_variables
        self.non_monotonic_vars = non_monotonic_vars or []
        self.positive_monotonic_vars = positive_monotonic_vars or []
        self.negative_monotonic_vars = negative_monotonic_vars or []

        # ---- build column index masks from names
        name_to_idx = {name: i for i, name in enumerate(self.all_variables)}
        self.mask_non = (torch.tensor([name_to_idx[n] for n in self.non_monotonic_vars], dtype=torch.long)
                         if self.non_monotonic_vars else None)
        self.mask_pos = (torch.tensor([name_to_idx[n] for n in self.positive_monotonic_vars], dtype=torch.long)
                         if self.positive_monotonic_vars else None)
        self.mask_neg = (torch.tensor([name_to_idx[n] for n in self.negative_monotonic_vars], dtype=torch.long)
                         if self.negative_monotonic_vars else None)

        # ---- create branches (Linear only; Tanh applied in forward)
        self.lin_non = (nn.Linear(len(self.non_monotonic_vars), hidden_non)
                        if self.non_monotonic_vars and hidden_non > 0 else None)
        self.lin_pos = (nn.Linear(len(self.positive_monotonic_vars), hidden_pos)
                        if self.positive_monotonic_vars and hidden_pos > 0 else None)
        self.lin_neg = (nn.Linear(len(self.negative_monotonic_vars), hidden_neg)
                        if self.negative_monotonic_vars and hidden_neg > 0 else None)

        total_h = ((self.lin_non.out_features if self.lin_non else 0) +
                   (self.lin_pos.out_features if self.lin_pos else 0) +
                   (self.lin_neg.out_features if self.lin_neg else 0))
        if total_h == 0:
            raise ValueError("At least one branch must have >0 hidden units.")

        self.out = nn.Linear(total_h, 1)  # final layer to 1 logit
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M] with columns in the order of self.all_variables
        returns: probabilities [B, 1]
        """
        branches = []
        # keep masks on same device as x
        if self.lin_non is not None:
            h_non = torch.tanh(self.lin_non(x.index_select(1, self.mask_non.to(x.device))))
            branches.append(h_non)
        if self.lin_pos is not None:
            h_pos = torch.tanh(self.lin_pos(x.index_select(1, self.mask_pos.to(x.device))))
            branches.append(h_pos)
        if self.lin_neg is not None:
            h_neg = torch.tanh(self.lin_neg(x.index_select(1, self.mask_neg.to(x.device))))
            branches.append(h_neg)

        h = branches[0] if len(branches) == 1 else torch.cat(branches, dim=1)
        logits = self.out(h)                 # [B, 1]
        probs = torch.sigmoid(logits)        # return probability for convenience
        return probs

    @torch.no_grad()
    def monotonic_constraint(self) -> None:
        """
        Project weights back to the feasible set after each optimizer step:
          - Positive branch first-layer weights: clamp to >= 0
          - Negative branch first-layer weights: clamp to <= 0
          - Output weights from pos/neg branches: clamp to >= 0
        Biases are unconstrained (do not affect derivative sign).
        """
        # First-layer constraints
        if self.lin_pos is not None:
            self.lin_pos.weight.clamp_(min=0.0)
        if self.lin_neg is not None:
            self.lin_neg.weight.clamp_(max=0.0)

        # Output-layer constraints per branch section
        start = 0
        # non-monotonic section (no constraint)
        if self.lin_non is not None:
            start += self.lin_non.out_features

        # positive-monotonic section: out weights >= 0
        if self.lin_pos is not None:
            end = start + self.lin_pos.out_features
            w = self.out.weight[:, start:end]
            w.clamp_(min=0.0)
            self.out.weight[:, start:end] = w
            start = end

        # negative-monotonic section: out weights >= 0
        if self.lin_neg is not None:
            end = start + self.lin_neg.out_features
            w = self.out.weight[:, start:end]
            w.clamp_(min=0.0)
            self.out.weight[:, start:end] = w
            # start = end  # not used further
