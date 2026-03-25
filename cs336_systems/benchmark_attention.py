from cs336_basics import *
from cs336_basics.model import *
from cs336_basics.loss import *


import torch
import time
import argparse
import yaml


# run forward + backward n_trials times and report avg/min runtime
def benchmark(cfg: dict):
    device = cfg["training"].get("device", "cuda")
    batch_size = cfg["training"]["batch_size"]
    model_cfg = cfg["model"]
    num_heads = model_cfg["num_heads"]
    seq_len = cfg["data"]["context_length"]
    d_head = model_cfg.get("d_head", model_cfg["d_model"] // num_heads)
    use_torch = cfg.get("use_flash", True)
    n_warmup = cfg.get("n_warmup", 3)
    n_trials = cfg.get("n_trials", 10)

    if use_torch:
        def attn_impl(Q, K, V, mask=None):
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    else:
        attn_impl = scaled_dot_product_attention

    def make_inputs():
        Q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, requires_grad=True)
        K = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, requires_grad=True)
        V = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, requires_grad=True)
        return Q, K, V

    def run_once():
        Q, K, V = make_inputs()
        out = attn_impl(Q, K, V)
        out.sum().backward()
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        run_once()

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000

    impl_name = "flash (torch)" if use_torch else "custom"
    print(f"Attention impl : {impl_name}")
    print(f"batch={batch_size}, heads={num_heads}, seq={seq_len}, d_head={d_head}")
    print(f"Avg fwd+bwd    : {avg_ms:.3f} ms")
    print(f"Min fwd+bwd    : {min_ms:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    benchmark(cfg)
