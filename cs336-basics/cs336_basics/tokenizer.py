from cs336_basics.bpe import train_bpe, pretokenize_chunk_text
from typing import Dict, List, Iterable, Iterator
import regex as re
import collections

import pickle

class Tokenizer: 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self, vocab, merges, special_tokens = None): 

        # tokenizer should have a table of 
        self.id_to_token = vocab 
        self.token_to_id = {v : k for k, v in vocab.items()}
        self.merges = merges 
        self.merge_ranks = collections.defaultdict(int)

        for i, merge in enumerate(merges): 
            self.merge_ranks[merge] = i
        self.special_tokens = special_tokens # these should already be in the id_to_token table 



    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
        vocab = None 
        merges = None 
        with open(vocab_filepath, "rb") as f: 
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f: 
            merges = pickle.load(f) 

        return Tokenizer(vocab, merges, special_tokens)



    def encode(self, text: str) -> list[int]: 

        tokenized = list()

        if self.special_tokens:
            # Split on special tokens so their characters are never pretokenized
            split_pattern = '(' + '|'.join(
                re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
            ) + ')'

            parts = [p for p in re.split(split_pattern, text) if p != '']
        else:
            parts = [text]

        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                tokenized.append(self.token_to_id[part.encode("utf-8")])
                continue

            for match in re.finditer(Tokenizer.PAT, part):
                token_bytes = match.group().encode("utf-8")
                word = tuple((b,) for b in token_bytes)   # start from single-byte symbols

                # merge this word according to the BPE merge rules 
                new_word = word 
                while True: 
                    
                    i = 0
                    best_merge = None 
                    best_merge_rank = float('inf')

                    while i < len(new_word):
                        if i < len(new_word) - 1:
                            pair_bytes = (bytes(new_word[i]), bytes(new_word[i + 1]))
                            if pair_bytes in self.merge_ranks:
                                if best_merge_rank > self.merge_ranks[pair_bytes]: 
                                    best_merge = (new_word[i], new_word[i+1])
                                    best_merge_rank = self.merge_ranks[pair_bytes]
                        i += 1

                    if not best_merge: 
                        break 
                    # else we execute that merge 
                    i = 0
                    new_word_next = list()
                    while i < len(new_word):
                        if i < len(new_word) - 1 and (new_word[i], new_word[i + 1]) == best_merge: 
                            merged_symbol = new_word[i] + new_word[i + 1]   # tuple concat
                            new_word_next.append(merged_symbol)
                            i += 2
                        else:
                            new_word_next.append(new_word[i])
                            i += 1

                    new_word = tuple(new_word_next) 
                # at this point, the word has been tokenized and we can add each token to the tokenized list 
                for token in new_word: 
                    tokenized.append(self.token_to_id[bytes(token)])
        return tokenized



        # first 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: 
        for text in iterable: 
            yield from self.encode(text)



    def decode(self, ids: list[int]) -> str: 
        # decode just reconstructs a list of token ids back into text 

        decoded_bytes = b""
        for id in ids: 
            decoded_bytes += self.id_to_token[id]
        
        return decoded_bytes.decode("utf-8", errors="replace")
        

        