import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


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
batch_size = 32
seq_len = 32
learning_rate = 1e-4
num_epochs = 10
num_samples = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2"))
model.to(device)
dataset = ShakespeareDataset(tokenizer, seq_len=256, max_samples=num_samples)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", total=len(train_dataloader))
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            eval_loss += output.loss.item()
    eval_loss /= len(val_dataloader) * batch_size

    # Log
    sample_input_ids = torch.tensor(
        tokenizer.encode("Hello, how are you?"), device=device
    )
    sample_text = tokenizer.decode(
        model.generate(sample_input_ids.reshape(1, -1), max_length=16)[0]
    )
    print(
        f"[Epoch {epoch+1}] Train loss: {loss.item():.2f} | Val loss: {eval_loss:.2f} | Sample: {repr(sample_text)}"
    )

    # Save checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
            "eval_loss": eval_loss,
            "sample_text": sample_text,
        },
        f"checkpoints/gpt2-latest.pth",
    )
