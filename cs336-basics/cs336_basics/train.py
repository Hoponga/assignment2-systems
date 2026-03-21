from cs336_basics.optim import *
from cs336_basics.model import *
from cs336_basics.tokenizer import *
from cs336_basics.dataloader import *
from cs336_basics.bpe import *
from cs336_basics.loss import CrossEntropyLoss
import numpy as np
import math
import os
import yaml
import wandb
import torch
from tqdm import tqdm


def get_lr(step: int, max_lr: float, min_lr: float, warmup_iters: int, max_iters: int) -> float:
    if step < warmup_iters:
        return max_lr * step / warmup_iters
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def run_train(model, train_dataset, val_dataset, optimizer, config, start_iter=0):
    train_cfg      = config["training"]
    opt_cfg        = config["optimizer"]
    sched_cfg      = config["lr_schedule"]
    ckpt_cfg       = config["checkpoint"]
    data_cfg       = config["data"]

    device         = str(train_cfg["device"])
    batch_size     = int(train_cfg["batch_size"])
    max_iters      = int(train_cfg["max_iters"])
    val_every      = int(train_cfg["val_every"])
    val_iters      = int(train_cfg["val_iters"])
    context_length = int(data_cfg["context_length"])
    grad_clip      = float(opt_cfg["grad_clip"])
    max_lr         = float(opt_cfg["lr"])
    min_lr         = float(sched_cfg["min_lr"])
    warmup_iters   = int(sched_cfg["warmup_iters"])
    save_every     = int(ckpt_cfg["save_every"])
    ckpt_dir       = str(ckpt_cfg["out_dir"])

    os.makedirs(ckpt_dir, exist_ok=True)
    model.to(device)
    criterion = CrossEntropyLoss()
    wandb.watch(model, log="all", log_freq=100)

    pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters)
    for step in pbar:
        # LR schedule
        lr = get_lr(step, max_lr, min_lr, warmup_iters, max_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Training step
        model.train()
        x, y = get_datapoints_from_source(train_dataset, batch_size, context_length, device)

        optimizer.zero_grad()
        logits = model(x)                                        # (B, T, vocab_size)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            run_gradient_clipping(model.parameters(), grad_clip)

        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        wandb.log({"train/loss": loss.item(), "lr": lr}, step=step)

        # Validation
        if (step + 1) % val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(val_iters):
                    vx, vy = get_datapoints_from_source(val_dataset, batch_size, context_length, device)
                    vlogits = model(vx)
                    val_losses.append(criterion(vlogits, vy).item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"step {step+1}/{max_iters} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | lr={lr:.2e}")
            wandb.log({"val/loss": val_loss}, step=step)

        # Checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path, wandb_run_id=wandb.run.id)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    with open("cs336_basics/configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device         = str(config["training"]["device"])
    vocab_size     = int(config["model"]["vocab_size"])
    checkpoint_src = config["checkpoint"]["resume_from"]  # str or None

    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    train_dataset = load_dataset_mmap(config["data"]["train_file"], dtype=dtype)
    val_dataset   = load_dataset_mmap(config["data"]["val_file"],   dtype=dtype)
    model = init_model_from_config(config)

    opt_cfg = config["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        betas=tuple(float(b) for b in opt_cfg["betas"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        eps=float(opt_cfg["eps"]),
    )

    start_iter = 0
    wandb_run_id = None
    if checkpoint_src is not None:
        start_iter, wandb_run_id = load_checkpoint(checkpoint_src, model, optimizer)

    if wandb_run_id is not None:
        wandb.init(
            project=config["wandb"]["project"],
            id=wandb_run_id,
            resume="must",
        )
    else:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
        )

    run_train(model, train_dataset, val_dataset, optimizer, config, start_iter)
