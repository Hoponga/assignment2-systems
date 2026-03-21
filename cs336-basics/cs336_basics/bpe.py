import collections
import multiprocessing as mp
from typing import Dict, List, Tuple
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


Symbol = Tuple[int, ...]
Word = Tuple[Symbol, ...]


def pretokenize_chunk_text(chunk: str, special_tokens: List[str] = None) -> Dict[Word, int]:

    token_counts = collections.defaultdict(int)

    if special_tokens:
        # Split on special tokens so their characters are never pretokenized
        split_pattern = '|'.join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        parts = re.split(split_pattern, chunk)
    else:
        parts = [chunk]

    for part in parts:
        for match in re.finditer(PAT, part):
            token_bytes = match.group().encode("utf-8")
            word = tuple((b,) for b in token_bytes)   # start from single-byte symbols
            token_counts[word] += 1

    return token_counts



# merge counts across chunks 
def merge_count_dicts(dicts: List[Dict[Word, int]]) -> Dict[Word, int]:

    merged = collections.defaultdict(int)
    for d in dicts:
        for word, count in d.items():
            merged[word] += count
    return merged


# given count dict then count the adjacent symbol pair frequencies in every word 
def get_stats(vocab: Dict[Word, int]) -> Dict[Tuple[Symbol, Symbol], int]:
    pairs = collections.defaultdict(int)

    for word, count in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += count

    return pairs


# replace every occurence of the pair with its fused version 
def merge_vocab(pair: Tuple[Symbol, Symbol], v_in: Dict[Word, int]) -> Dict[Word, int]:

    v_out = collections.defaultdict(int)

    for word, count in v_in.items():
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                merged_symbol = word[i] + word[i + 1]   # tuple concat
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        v_out[tuple(new_word)] += count

    return v_out


# chunk for multiprocessing — returns byte offsets, not loaded strings
def _chunk_file_for_pretokenization(
    input_path: str,
    num_processes: int,
    split_special_token: bytes,
) -> List[Tuple[int, int]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _pretokenize_file_slice(
    args: Tuple[str, int, int, List[str]],
) -> Dict[Word, int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_chunk_text(chunk, special_tokens)



# 336 spec takes in an input file path, final desired vocab size, list of special tokens, 
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 1,
    split_special_token: bytes = b"<|endoftext|>",
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes]
            Mapping token_id -> token bytes
        merges: list[tuple[bytes, bytes]]
            Merge rules in creation order
    """
    # Deduplicate special tokens while preserving order
    special_tokens = list(dict.fromkeys(special_tokens))

    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} is too small. "
            f"Need at least 256 + len(special_tokens) = {min_vocab_size}."
        )

    # each merge adds one token to the vocab, so the number of allowed merges is given by the total vocab size - the initial vocab size 
    num_merges = vocab_size - 256 - len(special_tokens)

    if num_processes <= 1:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        vocab_counts = pretokenize_chunk_text(text, special_tokens)
    else:
        offsets = _chunk_file_for_pretokenization(
            input_path=input_path,
            num_processes=num_processes,
            split_special_token=split_special_token,
        )

        args = [(input_path, start, end, special_tokens) for start, end in offsets]
        with mp.Pool(processes=len(offsets)) as pool:
            chunk_dicts = pool.map(_pretokenize_file_slice, args)

        vocab_counts = merge_count_dicts(chunk_dicts)


    merges: List[Tuple[bytes, bytes]] = []
    learned_tokens: List[bytes] = []

    for _ in range(num_merges):
        pair_stats = get_stats(vocab_counts)
        if not pair_stats:
            break

        #  tie-break:
        # highest frequency first, then lexicographically largest pair
        best_pair = max(pair_stats.items(), key=lambda kv: (kv[1], kv[0]))[0]

        left_sym, right_sym = best_pair
        left_bytes = bytes(left_sym)
        right_bytes = bytes(right_sym)
        merged_bytes = bytes(left_sym + right_sym)

        # spec wants a list of merges made 
        merges.append((left_bytes, right_bytes))
        learned_tokens.append(merged_bytes)

        vocab_counts = merge_vocab(best_pair, vocab_counts)


    vocab: Dict[int, bytes] = {}

    token_id = 0

    # base byte vocabulary 
    for b in range(256):
        vocab[token_id] = bytes([b])
        token_id += 1

    # special token vocabulary
    for tok in special_tokens:
        vocab[token_id] = tok.encode("utf-8")
        token_id += 1

    # learned merge tokens in merge order 
    for tok_bytes in learned_tokens:
        vocab[token_id] = tok_bytes
        token_id += 1

    return vocab, merges