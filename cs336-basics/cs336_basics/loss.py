import torch
import torch.nn as nn


class LossFunction(nn.Module):
    """Base class for loss functions. Subclasses must implement forward."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, T, vocab_size) — raw model outputs
            targets: (B, T)             — ground-truth token IDs
        Returns:
            Scalar loss.
        """
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    """
    Cross-entropy loss implemented from scratch via log-sum-exp.

    CE(x, y) = -x[y] + log(sum(exp(x)))

    We use the log-sum-exp trick for numerical stability:
        log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten sequence dimension: (B, T, V) → (N, V), (B, T) → (N,)
        N = logits.size(0) * logits.size(1)
        V = logits.size(-1)
        logits  = logits.view(N, V)   # (N, V)
        targets = targets.view(N)     # (N,)

        # Numerically stable log-softmax via log-sum-exp trick
        shift = logits.max(dim=-1, keepdim=True).values          # (N, 1)
        log_sum_exp = shift.squeeze(-1) + (logits - shift).exp().sum(dim=-1).log()  # (N,)

        # Gather the logit of the correct class for each token
        correct_logits = logits[torch.arange(N, device=logits.device), targets]     # (N,)

        # CE = -log_softmax[target] = -(correct_logit - log_sum_exp)
        loss = -(correct_logits - log_sum_exp)

        return loss.mean()
