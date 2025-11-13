import torch
from tqdm import tqdm
from spflow.modules.rat import RatSPN
from spflow.modules.leaf import Categorical
from spflow.meta import Scope
from spflow import log_likelihood

torch.manual_seed(0)

num_features = seq_len
K = len(tokenizer)
batch_size = 256

scope = Scope(list(range(num_features)))

leaf_layer = Categorical(
    scope=scope,
    out_channels=4,
    num_repetitions=2,
    K=K,
)

model = RatSPN(
    leaf_modules=[leaf_layer],
    n_root_nodes=1,
    n_region_nodes=8,
    num_repetitions=2,
    depth=3,
    outer_product=False,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
n_epochs = 100
log_every = 10
for epoch in range(n_epochs):
    for step, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
        optimizer.zero_grad()
        data = batch["input_ids"]
        ll = log_likelihood(model, data)  # (B,)
        loss = -ll.mean()  # NLL
        loss.backward()
        optimizer.step()
    if epoch % log_every == 0:
        print(f"[Epoch {epoch+1}] Loss {loss.item():.2f}")
