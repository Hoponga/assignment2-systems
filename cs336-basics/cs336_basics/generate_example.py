import pickle
import yaml
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

CHECKPOINT       = "checkpoints/checkpoint_20000.pt"
CONFIG           = "cs336_basics/configs/train_config.yaml"
VOCAB_MERGES_PKL = "../data/vocab_merges.pkl"
PROMPT           = "Once upon a time there was a little girl"

with open(CONFIG) as f:
    config = yaml.safe_load(f)

device = "mps"





with open(VOCAB_MERGES_PKL, "rb") as f:
    cached = pickle.load(f)



tokenizer = Tokenizer(cached["vocab"], cached["merges"], special_tokens=cached["special_tokens"])
eos_id = tokenizer.token_to_id.get("<|endoftext|>".encode("utf-8"))

# Load model
model = TransformerLM.from_pretrained(CHECKPOINT, config, device=device)

prompt_ids = tokenizer.encode(PROMPT)
prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

# Generate
out = model.generate(
    prompt_tensor,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    eos_token_id=eos_id,
)

generated_ids = out[0].tolist()
print(tokenizer.decode(generated_ids))
