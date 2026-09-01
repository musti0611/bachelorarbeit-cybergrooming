"""
GRUENDLICHE Pruefung der gesamten Pipeline: Annotation -> CSV -> BERT-Eingabe.
Rein lesend, kein Training. Laeuft auf CPU in ~1 Min.
"""

import sys, csv, re, collections
import pandas as pd

csv.field_size_limit(2**31 - 1)  # Segmente sind sehr lang

CSV_DIR = r"..\eSPD-datasets\PAN12\csv"
BEHAVIOR_TOKENS = [
    "[NEUTRAL]", "[VERTRAUENSAUFBAU]", "[INFORMATIONSGEWINNUNG]",
    "[GEHEIMHALTUNG_ISOLATION]", "[SEXUALISIERUNG]",
    "[KONTROLLE_NOETIGUNG]", "[OFFLINE_ESKALATION]",
]

def load(name):
    return pd.read_csv(f"{CSV_DIR}\\{name}", encoding="utf-8", engine="python")

print("="*80)
print("1) CSV-STRUKTUR & LABEL-VERTEILUNG")
print("="*80)
files = {
    "baseline train": "PAN12-train.csv", "baseline test": "PAN12-test.csv",
    "labeled  train": "PAN12labeled-train.csv", "labeled  test": "PAN12labeled-test.csv",
}
dfs = {}
for k, fn in files.items():
    df = load(fn); dfs[k] = df
    vc = df["label"].value_counts().to_dict()
    print(f"{k:16} | Zeilen: {len(df):>7,} | Spalten: {list(df.columns)} | {vc}")

print("\n" + "="*80)
print("2) FAIRNESS-CHECK: sind baseline & labeled 1:1 dieselben Konversationen?")
print("="*80)
for split in ["train", "test"]:
    b, l = dfs[f"baseline {split}"], dfs[f"labeled  {split}"]
    same_names = (b["chatName"].tolist() == l["chatName"].tolist())
    same_lbls  = (b["label"].tolist() == l["label"].tolist())
    print(f"{split}: gleiche Reihenfolge chatName? {same_names} | gleiche Labels? {same_lbls} | "
          f"Zeilen gleich? {len(b)==len(l)}")

print("\n" + "="*80)
print("3) TOKEN-CHECK: welche [LABEL]-Tokens stehen in der labeled-CSV?")
print("="*80)
tok_re = re.compile(r"\[[A-Z_]+\]")
found = collections.Counter()
for seg in dfs["labeled  train"]["segment"]:
    found.update(tok_re.findall(str(seg)))
print("Gefundene Tokens (train):")
for t, c in found.most_common():
    mark = "OK" if t in BEHAVIOR_TOKENS else "!! UNERWARTET"
    print(f"   {t:28} {c:>10,}  {mark}")
missing = [t for t in BEHAVIOR_TOKENS if t not in found]
print(f"Erwartete Tokens fehlen: {missing if missing else 'keine -> alle 7 vorhanden'}")

print("\n" + "="*80)
print("4) VALIDITAETS-CHECK: kommen Grooming-Labels in PREDATOR-Chats haeufiger vor?")
print("   (wenn ja, misst die Annotation echtes Signal)")
print("="*80)
for split in ["train"]:
    l = dfs[f"labeled  {split}"]
    per_class = {"predator": collections.Counter(), "non-predator": collections.Counter()}
    totals = {"predator": 0, "non-predator": 0}
    for lbl, seg in zip(l["label"], l["segment"]):
        toks = tok_re.findall(str(seg))
        per_class[lbl].update(toks)
        totals[lbl] += len(toks)
    print(f"[{split}] Anteil je Label an allen Nachrichten der Klasse:")
    print(f"   {'Label':28} {'predator':>12} {'non-predator':>14}")
    for t in BEHAVIOR_TOKENS:
        p = per_class['predator'][t]   / max(totals['predator'],1)
        n = per_class['non-predator'][t]/ max(totals['non-predator'],1)
        print(f"   {t:28} {p:>11.1%} {n:>13.1%}")

print("\n" + "="*80)
print("5) BILD DER EINGABE: so sieht eine echte Nachricht fuer BERT aus")
print("="*80)
from transformers import BertTokenizerFast
tk = BertTokenizerFast.from_pretrained("bert-base-uncased")
tk_lab = BertTokenizerFast.from_pretrained("bert-base-uncased")
tk_lab.add_special_tokens({"additional_special_tokens": BEHAVIOR_TOKENS})

# nimm eine Predator-Konversation
lab_train = dfs["labeled  train"]; base_train = dfs["baseline train"]
idx = lab_train.index[lab_train["label"] == "predator"][0]
base_seg = str(base_train.loc[idx, "segment"])
lab_seg  = str(lab_train.loc[idx, "segment"])

print(f"\nBASELINE Text (Anfang):\n  {base_seg[:120]!r}")
print(f"LABELED  Text (Anfang):\n  {lab_seg[:150]!r}")

print("\n-- Tokenisierung des labeled-Textes (erste 22 Tokens) --")
enc = tk_lab(lab_seg[:200])
toks = tk_lab.convert_ids_to_tokens(enc["input_ids"])[:22]
print("  ", toks)

vt_id = tk_lab.convert_tokens_to_ids("[VERTRAUENSAUFBAU]")
print(f"\n[VERTRAUENSAUFBAU] MIT special-token  -> EIN Token, id={vt_id}")
print(f"[VERTRAUENSAUFBAU] OHNE special-token -> zerstueckelt in: "
      f"{tk.tokenize('[VERTRAUENSAUFBAU]')}")

print("\n-- Truncation-Hinweis: wie viele Tokens hat die volle Konversation? --")
full_base = len(tk(base_seg)["input_ids"])
full_lab  = len(tk_lab(lab_seg)["input_ids"])
print(f"  baseline: {full_base:,} Tokens | labeled: {full_lab:,} Tokens  (max_len=512 schneidet ab)")
