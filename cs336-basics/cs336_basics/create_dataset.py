"""
Tokenize raw text files using a trained BPE tokenizer and save as .npy arrays.
The saved files can be loaded with np.load(..., mmap_mode='r') for memory-mapped
access during training without loading the full dataset into RAM.
"""

import numpy as np
import os
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def tokenize_file_to_npy(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
    dtype: np.dtype = np.uint16,
) -> None:
    """
    Tokenize a raw text file and save the token IDs as a flat .npy array.

    Args:
        input_path:  Path to the raw .txt file.
        output_path: Destination .npy file path.
        tokenizer:   A fitted Tokenizer instance.
        dtype:       Integer dtype for the token IDs (uint16 supports vocab ≤ 65535,
                     use uint32 for larger vocabularies).
    """
    print(f"Tokenizing {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    arr = np.array(token_ids, dtype=dtype)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, arr)

    # Sanity-check: reload and verify no out-of-range token IDs
    check = np.load(output_path, mmap_mode="r")
    vocab_size = len(tokenizer.id_to_token)
    assert int(check.max()) < vocab_size, (
        f"Token ID {int(check.max())} exceeds vocab_size {vocab_size}"
    )
    assert int(check.min()) >= 0, "Negative token IDs found"
    print(f"Sanity check passed: values in [0, {int(check.max())}], vocab_size={vocab_size}")



    

if __name__ == "__main__":
    import yaml

    with open("cs336_basics/configs/train_config_create_dataset.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_txt = config["data"]["train_file"]
    val_txt   = config["data"]["val_file"]
    vocab_size = config["model"]["vocab_size"]
    special_tokens = config["data"]["special_tokens"]

    # output npy files are just strings of tokens 
    train_npy = os.path.splitext(train_txt)[0] + ".npy"
    val_npy   = os.path.splitext(val_txt)[0]   + ".npy"

    print("Training BPE ...")
    vocab, merges = train_bpe(
        train_txt, vocab_size, special_tokens=special_tokens, num_processes=64
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)

    # Choose dtype based on vocab size
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    # print("tokenizing train file")
    # tokenize_file_to_npy(train_txt, train_npy, tokenizer, dtype=dtype)

    print("tokenizing val file")
    tokenize_file_to_npy(val_txt,   val_npy,   tokenizer, dtype=dtype)

