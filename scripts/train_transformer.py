from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, top_k_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
OUT = ROOT / "models" / "transformer_pose_hands_graph_run"
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")

with open(META / "label_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)

def load_split(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        arr = np.load(row["keypoint_path"], allow_pickle=False)
        # Original pose + hands: 258 features across 30 frames.
        X.append(arr[:, 1404:1662].astype(np.float32))
        y.append(mapping[str(row["gloss"])])

    return np.stack(X), np.asarray(y, dtype=np.int64)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

print("Loading signer-disjoint splits...")
X_train, y_train = load_split(META / "train_signer_disjoint.csv")
X_val, y_val = load_split(META / "validation_signer_disjoint.csv")
print("Train:", X_train.shape, "| Validation:", X_val.shape)

# Standardize each feature using training data only.
mean = X_train.reshape(-1, 258).mean(axis=0)
std = X_train.reshape(-1, 258).std(axis=0)
std[std < 1e-6] = 1.0
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

train_loader = DataLoader(
    SequenceDataset(X_train, y_train), batch_size=32, shuffle=True
)
val_loader = DataLoader(
    SequenceDataset(X_val, y_val), batch_size=64, shuffle=False
)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=30):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=258, dim=128, heads=4,
                 layers=2, classes=100):
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)
        self.position = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=256,
            dropout=0.2, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(dim, classes)
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

model = TransformerClassifier(classes=len(mapping)).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0003, weight_decay=1e-3
)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

best_loss = float("inf")
best_state = None
stale = 0
patience = 8
history = []

print("\nTraining Transformer...")
for epoch in range(1, 41):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_count = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * len(yb)
        train_correct += (logits.argmax(1) == yb).sum().item()
        train_count += len(yb)

    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_logits.append(model(xb))
            val_labels.append(yb)

    logits = torch.cat(val_logits)
    labels = torch.cat(val_labels)
    val_loss = criterion(logits, labels).item()
    val_acc = (logits.argmax(1) == labels).float().mean().item()
    val_probs = logits.softmax(dim=1)
    val_top5 = top_k_accuracy_score(
        y_val, val_probs.numpy(), k=5, labels=np.arange(len(mapping))
    )
    scheduler.step(val_loss)

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "validation_loss": val_loss,
            "validation_top1_accuracy": val_acc,
            "validation_top5_accuracy": val_top5,
        }
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss/train_count:.4f} | "
        f"train_acc={train_correct/train_count:.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        stale = 0
    else:
        stale += 1

    if stale >= patience:
        print("Early stopping.")
        break

history_path = OUT / "training_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

model.load_state_dict(best_state)
model.eval()
all_logits = []
with torch.no_grad():
    for xb, _ in val_loader:
        all_logits.append(model(xb))

probs = torch.cat(all_logits).softmax(dim=1).numpy()
top1 = accuracy_score(y_val, probs.argmax(axis=1))
top5 = top_k_accuracy_score(
    y_val, probs, k=5, labels=np.arange(len(mapping))
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": 258,
        "dim": 128,
        "heads": 4,
        "layers": 2,
        "classes": len(mapping),
        "mean": mean,
        "std": std,
    },
    OUT / "transformer_pose_hands.pt"
)

print("\n===== TRANSFORMER RESULTS =====")
print(f"Top-1 accuracy: {top1:.4f} ({top1*100:.2f}%)")
print(f"Top-5 accuracy: {top5:.4f} ({top5*100:.2f}%)")
print("History saved to:", history_path)
print("Saved to:", OUT)
