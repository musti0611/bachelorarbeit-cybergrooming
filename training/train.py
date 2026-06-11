"""
BERT-based eSPD Trainer
Trains two variants:
  - baseline:  raw text only
  - with_labels: behavior labels prepended as special tokens

Usage:
  python train.py --variant baseline --data_dir ../eSPD-datasets/PAN12/csv --dataset PAN12
  python train.py --variant with_labels --data_dir ../eSPD-datasets/PANC/csv --dataset PANC
"""

import argparse
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# ── CLI args ────────────────────────────────────────────────────────────────
#Trainingsparameter die übergeben werden, z.B. welche Variante, lr(learning rate): wie groß die Anpassungsschritte sind. 
parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["baseline", "with_labels"], required=True)
parser.add_argument("--data_dir", required=True, help="Directory containing train/test CSVs")
parser.add_argument("--dataset", required=True, help="Dataset name, e.g. PAN12 or PANC")
parser.add_argument("--model_name", default="bert-base-uncased")
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--output_dir", default="models")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Variant: {args.variant}")

LABEL2ID = {"non-predator": 0, "predator": 1}
ID2LABEL = {0: "non-predator", 1: "predator"}

# Behavior labels used as special tokens
BEHAVIOR_TOKENS = [
    "[NEUTRAL]", "[VERTRAUENSAUFBAU]", "[INFORMATIONSGEWINNUNG]",
    "[GEHEIMHALTUNG_ISOLATION]", "[SEXUALISIERUNG]",
    "[KONTROLLE_NOETIGUNG]", "[OFFLINE_ESKALATION]",
]

# ── Dataset ─────────────────────────────────────────────────────────────────

#Die Klasse lädt die CSV und macht daraus etwas, das Pytorch verstehen kann. 
class ChatSegmentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["segment"].tolist()
        self.labels = [LABEL2ID[l] for l in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Load data ────────────────────────────────────────────────────────────────

train_path = os.path.join(args.data_dir, f"{args.dataset}-train.csv")
test_path  = os.path.join(args.data_dir, f"{args.dataset}-test.csv")

df_train = pd.read_csv(train_path, encoding="utf-8")
df_test  = pd.read_csv(test_path,  encoding="utf-8")

print(f"\nTrain: {len(df_train)} samples | Test: {len(df_test)} samples")
print("Train label dist:\n", df_train["label"].value_counts())

# ── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

#BERT Vokabular wird um meine Label erweitert, sodass er selber dann lernt was z.B. [VERTRAUENSAUFBAU] bedeutet 
if args.variant == "with_labels":
    tokenizer.add_special_tokens({"additional_special_tokens": BEHAVIOR_TOKENS})
    print(f"Added {len(BEHAVIOR_TOKENS)} behavior-label special tokens to tokenizer")

train_ds = ChatSegmentDataset(df_train, tokenizer, args.max_len)
test_ds  = ChatSegmentDataset(df_test,  tokenizer, args.max_len)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

# ── Model ────────────────────────────────────────────────────────────────────

model = BertForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

if args.variant == "with_labels":
    model.resize_token_embeddings(len(tokenizer))

model.to(DEVICE)

# ── Class weights (handle imbalance) ────────────────────────────────────────
# Im Datensatz gibt es viel mehr non predator als Predator. Wenn das Modell einfach Predator ratet, hat es eine hohe Erfolgswahrscheinlichkeit. 
# Deshalb machen wir class weights um zu sagen, einen Predator zu übersehen ist schlimmer als ein Fehlalarm
counts = df_train["label"].value_counts()
n_neg = counts.get("non-predator", 1)
n_pos = counts.get("predator", 1)
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(DEVICE)
print(f"\nClass weight for 'predator': {pos_weight.item():.2f}")

# ── Optimizer & Scheduler ────────────────────────────────────────────────────

optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ── Training loop ────────────────────────────────────────────────────────────

  #1. outputs = model(...) → Das Modell macht eine Vorhersage
  #2. loss = outputs.loss → Wie falsch war die Vorhersage? (Zahl zwischen 0 und ∞)
  #3. loss.backward() → Berechne, welche Gewichte schuld am Fehler sind (Backpropagation)
  #4. optimizer.step() → Passe die Gewichte an (einen kleinen Schritt in die richtige Richtung)
  #5. optimizer.zero_grad() → Reset für die nächste Runde

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=["non-predator", "predator"])
    metrics = {
        "f1":        f1_score(all_labels, all_preds, pos_label=1),
        "precision": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, pos_label=1),
    }
    return report, metrics


# ── Run ──────────────────────────────────────────────────────────────────────

out_dir = Path(args.output_dir) / f"{args.dataset}_{args.variant}"
out_dir.mkdir(parents=True, exist_ok=True)

best_f1 = 0
history = []

for epoch in range(1, args.epochs + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{args.epochs}")
    avg_loss = train_epoch(model, train_loader)
    report, metrics = evaluate(model, test_loader)

    print(f"Loss: {avg_loss:.4f}")
    print(report)
    print(f"F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")

    history.append({"epoch": epoch, "loss": avg_loss, **metrics})

    # Nach jeder Epoche wird das Modell auf den Testdaten evaluiert. Das Modell mit dem besten F1-Score wird gespeichert. So hat man am Ende nicht das "letzte" Modell, sondern das "beste".
    if metrics["f1"] >= best_f1:
        best_f1 = metrics["f1"]
        model.save_pretrained(str(out_dir / "best_model"))
        tokenizer.save_pretrained(str(out_dir / "best_model"))
        print(f"  -> Saved best model (F1={best_f1:.4f})")

# ── Save results ─────────────────────────────────────────────────────────────

with open(out_dir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nDone. Best F1: {best_f1:.4f}")
print(f"Model saved to: {out_dir / 'best_model'}")
