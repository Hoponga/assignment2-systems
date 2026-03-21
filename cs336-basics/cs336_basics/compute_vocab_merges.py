"""
Run this once to train BPE and cache vocab/merges to disk.
Usage: python cs336_basics/compute_vocab_merges.py
Output: data/vocab_merges.pkl (path configurable via CONFIG / OUTPUT below)
"""
import os
import pickle
import yaml
from cs336_basics.bpe import train_bpe

CONFIG = "cs336_basics/configs/train_config_create_dataset.yaml"
OUTPUT = "../data/vocab_merges.pkl"

if __name__ == "__main__":
    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    vocab_size = int(config["model"]["vocab_size"])
    special_tokens = data_cfg["special_tokens"]
    train_file = data_cfg["train_file"]

    print(f"Training BPE on {train_file} (vocab_size={vocab_size}) ...")
    vocab, merges = train_bpe(
        train_file,
        vocab_size,
        special_tokens=special_tokens,
        num_processes=64,
    )

    with open(OUTPUT, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)

    print(f"Saved vocab/merges to {OUTPUT}")
