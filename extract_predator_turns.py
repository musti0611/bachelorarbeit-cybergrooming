"""
Extrahiert Predator-Turns aus dem PAN12-Korpus für manuelle KI-Annotation.
Ziel: diverse Sample über alle Predators, min. Textlänge, kein Duplikat-Spam.
"""

import xml.etree.ElementTree as ET
import html
import random

XML_PATHS = [
    r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-2012-05-01.xml",
    r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-test-corpus-2012-05-17.xml",
]
PRED_PATH = r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
OUT_PATH  = r"C:\Users\mugur\Bachelorarbeit\predator_turns_sample.txt"

MIN_LEN = 30          # Mindestlänge in Zeichen (filtert "lol", "ok" etc.)
MAX_PER_PREDATOR = 40 # max Turns pro Predator → Diversität über viele Predators
random.seed(42)

print("Lade Predator-Liste ...")
with open(PRED_PATH, encoding="utf-8") as f:
    predators = set(l.strip() for l in f if l.strip())
print(f"  {len(predators)} Predators.")

# Sammle alle Turns pro Predator
turns_by_predator: dict[str, list[tuple[str, str]]] = {p: [] for p in predators}

for xml_path in XML_PATHS:
    name = xml_path.split("\\")[-1]
    print(f"Parse {name} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for conv in root.findall("conversation"):
        conv_id = conv.get("id")
        for msg in conv.findall("message"):
            author = msg.findtext("author") or ""
            if author not in predators:
                continue
            text = html.unescape(msg.findtext("text") or "").strip()
            if len(text) < MIN_LEN:
                continue
            turns_by_predator[author].append((conv_id, text))

# Sample: max MAX_PER_PREDATOR zufällige Turns pro Predator
all_turns = []
for predator, turns in turns_by_predator.items():
    random.shuffle(turns)
    sample = turns[:MAX_PER_PREDATOR]
    for conv_id, text in sample:
        all_turns.append((predator, conv_id, text))

random.shuffle(all_turns)
print(f"\nGesamte Turns im Sample: {len(all_turns)}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, (pred, conv_id, text) in enumerate(all_turns, 1):
        f.write(f"[{i:04d}] CONV:{conv_id} | PRED:{pred[:16]}\n")
        f.write(f"       {text[:400]}\n")
        f.write("\n")

print(f"Sample gespeichert: {OUT_PATH}")
