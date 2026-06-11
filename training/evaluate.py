"""
eSPD Prefix-Based Evaluation Script
====================================
Evaluiert ein trainiertes BERT-Modell auf die eSPD-spezifische Art:
  - Prefix-basiert: nach jeder neuen Nachricht eine Vorhersage
  - MasterClassifier mit Skeptizismus s=1..10
  - Metriken: Precision, Recall, F1, Warnlatenz, F-latency

Verwendung:
  python evaluate.py \
    --model_dir models/PANC_baseline/best_model \
    --datapack ../eSPD-datasets/PAN12/datapacks/datapack-PAN12-test.json \
    --output_dir results/PANC_baseline \
    --variant baseline

  Für das erweiterte Modell:
  python evaluate.py \
    --model_dir models/PANC_with_labels/best_model \
    --datapack ../eSPD-datasets/PANC/datapacks/datapack-PANC-test.json \
    --output_dir results/PANC_with_labels \
    --variant with_labels

Referenz: Vogt, Leser, Akbik (2021). Early Detection of Sexual Predators in Chats. ACL 2021.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from tqdm import tqdm

# ── CLI ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",   required=True,  help="Pfad zum gespeicherten BERT-Modell")
parser.add_argument("--datapack",    required=True,  help="Pfad zum Test-Datapack (JSON)")
parser.add_argument("--output_dir",  required=True,  help="Ausgabeverzeichnis für Ergebnisse")
parser.add_argument("--variant",     required=True,  choices=["baseline", "with_labels"])
parser.add_argument("--window_size", type=int, default=50,  help="Anzahl Nachrichten im Sliding Window")
parser.add_argument("--max_len",     type=int, default=512, help="Max. Token-Länge für BERT")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Modell: {args.model_dir}")
print(f"Variante: {args.variant}")

# ── Modell laden ─────────────────────────────────────────────────────────────

import os, json as _json
model_dir_abs = str(Path(args.model_dir).resolve())

# Lade Tokenizer direkt aus lokalen Dateien (umgeht HuggingFace-Hub-Validierung auf Windows)
tokenizer = BertTokenizerFast(
    vocab_file=os.path.join(model_dir_abs, "vocab.txt"),
    tokenizer_config_file=os.path.join(model_dir_abs, "tokenizer_config.json"),
)
# Sonderfall: zusätzliche Special Tokens (Behavior-Labels) laden falls vorhanden
special_tokens_path = os.path.join(model_dir_abs, "special_tokens_map.json")
if os.path.exists(special_tokens_path):
    with open(special_tokens_path) as _f:
        _stm = _json.load(_f)
    if "additional_special_tokens" in _stm:
        tokenizer.add_special_tokens({"additional_special_tokens": _stm["additional_special_tokens"]})

# Lade Modell: Config + Weights direkt
from transformers import BertConfig
config = BertConfig.from_json_file(os.path.join(model_dir_abs, "config.json"))
model = BertForSequenceClassification(config)
weights_path = os.path.join(model_dir_abs, "model.safetensors")
if os.path.exists(weights_path):
    from safetensors.torch import load_file as load_safetensors
    state_dict = load_safetensors(weights_path)
else:
    state_dict = torch.load(os.path.join(model_dir_abs, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ── Hilfsfunktionen (nach Vogt et al. 2021) ──────────────────────────────────

def is_nonempty_msg(msg):
    """Nachricht ist nicht leer und ist vom Typ 'message'."""
    return (
        msg is not None
        and msg.get("type") == "message"
        and bool(msg.get("body", "").strip())
    )

def messages_to_text(messages):
    """Verkette Nachrichten zu einem String (Sliding Window → BERT-Input)."""
    return " ".join(m["body"] for m in messages if is_nonempty_msg(m))

def predict_predator_prob(text):
    """Gibt die Wahrscheinlichkeit zurück, dass der Text von einem Predator stammt."""
    enc = tokenizer(
        text,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=1)

    return probs[0][1].item()  # Wahrscheinlichkeit für Label "predator" (Index 1)


class MasterClassifier:
    """
    Skeptizismus-basierter Klassifikator nach Vogt et al. (2021).
    Löst Alarm aus, wenn mindestens 'skepticism' der letzten 10
    Vorhersagen positiv (predator) waren.
    """
    WINDOW = 10

    def __init__(self, skepticism):
        self.skepticism = skepticism
        self.history = [False] * self.WINDOW

    def add_prediction(self, is_dangerous: bool) -> bool:
        if self.state_is_dangerous():
            return True  # einmal Alarm = immer Alarm
        self.history.insert(0, is_dangerous)
        self.history.pop()
        return self.state_is_dangerous()

    def state_is_dangerous(self) -> bool:
        return sum(self.history) >= self.skepticism


class Score:
    """Berechnet Precision, Recall, F1 für die Predator-Klasse."""

    def __init__(self):
        self.tp = self.fp = self.tn = self.fn = 0

    def update(self, predicted_positive: bool, is_positive: bool):
        if     is_positive and     predicted_positive: self.tp += 1
        elif   is_positive and not predicted_positive: self.fn += 1
        elif not is_positive and not predicted_positive: self.tn += 1
        elif not is_positive and     predicted_positive: self.fp += 1

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def __repr__(self):
        return (f"Score(P={self.precision:.4f}, R={self.recall:.4f}, "
                f"F1={self.f1:.4f}, tp={self.tp}, fp={self.fp}, "
                f"fn={self.fn}, tn={self.tn})")


def compute_speed(latencies, median_half_penalty=90):
    """
    Speed-Metrik nach Vogt et al. (2021):
    Misst wie früh Warnungen kommen. Speed=1 → sofortige Warnung.
    Speed=0 → Warnung nach median_half_penalty Nachrichten.
    """
    if not latencies:
        return 0.0
    penalty_factor = np.log(3) / (median_half_penalty - 1)

    def penalty(delay):
        return -1 + 2 / (1 + np.exp(-penalty_factor * (delay - 1)))

    return float(1 - np.median([penalty(d) for d in latencies]))

# ── Datapack laden ────────────────────────────────────────────────────────────

print(f"\nLade Datapack: {args.datapack}")
with open(args.datapack, "r", encoding="utf-8") as f:
    datapack = json.load(f)

chats = datapack["chats"]
print(f"Chats gesamt: {len(chats)}")
print(f"Davon Predator-Chats: {sum(1 for c in chats.values() if c['className'] == 'predator')}")

# ── Prefix-Annotation ─────────────────────────────────────────────────────────
# Für jeden Chat: nach jeder Nachricht eine Vorhersage treffen.

print("\nAnnotiere Chats prefix-basiert...")
for chat_name, chat in tqdm(chats.items()):
    nonempty = [m for m in chat["content"] if is_nonempty_msg(m)]
    for i, msg in enumerate(nonempty):
        window = nonempty[max(0, i + 1 - args.window_size): i + 1]
        text   = messages_to_text(window)
        msg["prediction"] = predict_predator_prob(text)

# ── Evaluation für jeden Skeptizismus-Wert ────────────────────────────────────

print("\nEvaluiere für Skeptizismus s=1..10...")

results = []

for skepticism in range(1, 11):
    score     = Score()
    latencies = []

    for chat_name, chat in chats.items():
        is_positive  = (chat["className"] == "predator")
        nonempty     = [m for m in chat["content"] if is_nonempty_msg(m)]
        mc           = MasterClassifier(skepticism)
        warning_turn = None

        for i, msg in enumerate(nonempty):
            if "prediction" not in msg:
                continue
            raised = mc.add_prediction(msg["prediction"] >= 0.5)
            if raised and warning_turn is None:
                warning_turn = i + 1  # 1-basiert

        warned = mc.state_is_dangerous()
        score.update(warned, is_positive)

        # Latenz nur für korrekte True-Positive-Warnungen
        if is_positive and warned and warning_turn is not None:
            latencies.append(warning_turn)

    speed     = compute_speed(latencies)
    f_latency = score.f1 * speed

    results.append({
        "skepticism": skepticism,
        "precision":  round(score.precision, 4),
        "recall":     round(score.recall, 4),
        "f1":         round(score.f1, 4),
        "speed":      round(speed, 4),
        "f_latency":  round(f_latency, 4),
        "median_latency": int(np.median(latencies)) if latencies else None,
        "num_warnings": len(latencies),
        "tp": score.tp, "fp": score.fp, "fn": score.fn, "tn": score.tn,
    })

    print(f"  s={skepticism:2d} | P={score.precision:.4f}  R={score.recall:.4f}  "
          f"F1={score.f1:.4f}  speed={speed:.4f}  F-latency={f_latency:.4f}  "
          f"median_lat={np.median(latencies) if latencies else 'N/A'}")

# ── Ergebnisse speichern ──────────────────────────────────────────────────────

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

results_path = out / "espd_evaluation.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({"variant": args.variant, "results_by_skepticism": results}, f, indent=2)

print(f"\nErgebnisse gespeichert: {results_path}")

# ── Übersichtstabelle ─────────────────────────────────────────────────────────

print("\n" + "="*75)
print(f"{'Ergebnisse für Variante: ' + args.variant:^75}")
print("="*75)
print(f"{'s':>4} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'F-latency':>10} | {'Med.Lat':>8}")
print("-"*75)
for r in results:
    med = str(r['median_latency']) if r['median_latency'] else "N/A"
    print(f"{r['skepticism']:>4} | {r['precision']:>10.4f} | {r['recall']:>10.4f} | "
          f"{r['f1']:>10.4f} | {r['f_latency']:>10.4f} | {med:>8}")
print("="*75)

best = max(results, key=lambda r: r["f_latency"])
print(f"\nBestes F-latency bei s={best['skepticism']}: {best['f_latency']:.4f}")
