from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from spflow.meta import Scope
from spflow.modules.rat import RatSPN
from spflow.modules.leaf import Binomial, Categorical
from spflow import log_likelihood


# Define dataset
class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        seq_len=256,
        max_samples=None,
        file_path="data/shakespeare.txt",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Read Shakespeare text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Flatten to single sequence, tokeniz and reshape to (num_batches, seq_len)
        tokens = self.tokenizer.encode(text)
        n_batches = len(tokens) // seq_len
        self.sequences = torch.tensor(
            tokens[: n_batches * seq_len], dtype=torch.long
        ).reshape(n_batches, seq_len)
        if max_samples is not None:
            self.sequences = self.sequences[:max_samples]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"input_ids": seq, "labels": seq}


# 1. Shakespeare dataset
file_path = "../data/shakespeare/main.txt"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
seq_len = 8
batch_size = 512
max_samples = None

dataset = ShakespeareDataset(tokenizer, seq_len=seq_len, max_samples=max_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

depth = 3
n_region_nodes = 5
num_leaves = 5
num_repetitions = 2
n_root_nodes = 1
num_feature = seq_len
# n = torch.tensor(16) # total count for binomial distribution
n = torch.tensor(len(tokenizer))

scope = Scope(list(range(0, num_feature)))

# rat_leaf_layer = Binomial(scope=scope, n=n, out_channels=num_leaves, num_repetitions=num_repetitions)
rat_leaf_layer = Categorical(
    scope=scope, K=n, out_channels=num_leaves, num_repetitions=num_repetitions
)
rat = RatSPN(
    leaf_modules=[rat_leaf_layer],
    n_root_nodes=n_root_nodes,
    n_region_nodes=n_region_nodes,
    num_repetitions=num_repetitions,
    depth=depth,
    outer_product=True,
    split_halves=True,
)


optimizer = torch.optim.Adam(rat.parameters(), lr=1e-2)
n_epochs = 1_000
log_every = 500
for epoch in range(n_epochs):
    for step, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
        optimizer.zero_grad()
        data = batch["input_ids"]
        ll = log_likelihood(rat, data)  # (B,)
        loss = -ll.mean()  # NLL
        loss.backward()
        optimizer.step()
    if epoch % log_every == 0:
        print(f"[Epoch {epoch}] Loss {loss.item():.2f}")
