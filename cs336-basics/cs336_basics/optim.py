import torch

from typing import Optional, Callable, Iterable


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"INVALID LEARNING RATE {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']
                g = p.grad

                # Update biased moment estimates
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias-corrected estimates
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Weight decay (decoupled)
                p.data.mul_(1 - lr * weight_decay)

                # Parameter update
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)
