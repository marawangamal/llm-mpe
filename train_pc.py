from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from spflow.meta import Scope
from spflow.modules.rat import RatSPN
from spflow.modules.leaf import Binomial, Categorical
from spflow import log_likelihood, sample


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


# Hyperparameters
file_path = "../data/shakespeare/main.txt"
seq_len = 8
batch_size = 512
max_samples = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 1_000
log_every = 500

# Model HPs
depth = 3
n_region_nodes = 5
num_leaves = 5
num_repetitions = 2
n_root_nodes = 1
num_feature = seq_len

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = ShakespeareDataset(tokenizer, seq_len=seq_len, max_samples=max_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

scope = Scope(list(range(0, num_feature)))
# rat_leaf_layer = Binomial(scope=scope, n=torch.tensor(len(tokenizer)), out_channels=num_leaves, num_repetitions=num_repetitions)
rat_leaf_layer = Categorical(
    scope=scope,
    K=len(tokenizer),
    out_channels=num_leaves,
    num_repetitions=num_repetitions,
)
model_pc = RatSPN(
    leaf_modules=[rat_leaf_layer],
    n_root_nodes=n_root_nodes,
    n_region_nodes=n_region_nodes,
    num_repetitions=num_repetitions,
    depth=depth,
    outer_product=True,
    split_halves=True,
)
model_pc.to(device)

optimizer = torch.optim.Adam(model_pc.parameters(), lr=1e-2)


for epoch in range(n_epochs):
    pbar = tqdm(dataloader, leave=False, total=len(dataloader))
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        data = batch["input_ids"]
        ll = log_likelihood(model_pc, data)  # (B,)
        loss = -ll.mean()  # NLL
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    if epoch % log_every == 0:
        x_hat = sample(model_pc, 1)
        x_text = repr(tokenizer.decode(x_hat[0].to(torch.long)))
        print(f"[Epoch {epoch}] Loss {loss.item():.2f} | Sample: {x_text}")
