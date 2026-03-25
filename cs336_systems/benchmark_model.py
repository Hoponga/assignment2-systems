from cs336_basics import *
from cs336_basics.model import *
from cs336_basics.loss import *
import torch
import time
import argparse
import yaml


# run forward + backward 10 times and take average runtime
def benchmark(cfg: dict):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    cross_entropy = CrossEntropyLoss()

    device = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = training_cfg.get("batch_size", 8)
    context_length = data_cfg.get("context_length", 128)
    vocab_size = model_cfg.get("vocab_size", 10000)

    num_warmup = cfg.get("benchmark", {}).get("num_warmup", 1)
    num_iters = cfg.get("benchmark", {}).get("num_iters", 3)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    ).to(device)

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        logits = model(x)
        loss = cross_entropy(logits, labels)
        loss.backward()
        model.zero_grad()
        sync()

    # Timed iters
    fwd_times, bwd_times = [], []
    for _ in range(num_iters):
        sync()
        t0 = time.perf_counter()
        logits = model(x)
        loss = cross_entropy(logits, labels)
        sync()
        t1 = time.perf_counter()

        loss.backward()
        model.zero_grad()
        sync()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    fwd_ms = sum(fwd_times) / num_iters
    bwd_ms = sum(bwd_times) / num_iters

    print(num_iters)
    print(f"Forward:    {fwd_ms:.2f} ms/iter")
    print(f"Backward:  {bwd_ms:.2f} ms/iter")
    print(f"Forward+Backward: {fwd_ms + bwd_ms:.2f} ms/iter  "
          f"(batch={batch_size}, seq={context_length}, layers={model_cfg['num_layers']}, "
          f"d_model={model_cfg['d_model']}, device={device})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    benchmark(cfg)
