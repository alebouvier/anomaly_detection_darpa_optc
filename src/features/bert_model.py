import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class BertWithProjectionOnly(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.projection = nn.Linear(768, output_dim)

    def forward(self, **inputs):
        outputs = self.bert(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.projection(pooled)


def contrastive_loss(embeddings: torch.Tensor, temperature=0.05) -> torch.Tensor:
    """
    SimCSE-style loss: each log is passed twice (with dropout as augmentation),
    and the two versions of the same log are treated as positive pairs.
    """
    embeddings = nn.functional.normalize(embeddings, dim=-1)
    # Split into two augmented halves
    half = embeddings.shape[0] // 2
    e1, e2 = embeddings[:half], embeddings[half:]

    similarity = torch.matmul(e1, e2.T) / temperature   # [batch/2, batch/2]
    labels = torch.arange(half)                          # diagonal = positives
    loss = nn.CrossEntropyLoss()(similarity, labels)
    return loss


def train_unsupervised(logs, output_dim=128, epochs=5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertWithProjectionOnly(output_dim=output_dim)

    # Freeze BERT
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=1e-3)

    model.train()  # Keep dropout ON — it acts as augmentation
    for epoch in range(epochs):
        # Duplicate each log to form positive pairs
        doubled = logs + logs
        inputs = tokenizer(doubled, return_tensors="pt",
                           truncation=True, padding=True, max_length=512)
        optimizer.zero_grad()
        embeddings = model(**inputs)
        loss = contrastive_loss(embeddings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} — Loss: {loss.item():.4f}")

    return model