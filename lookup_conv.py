"""
Sucht eine Conversation im PAN12-Trainingskorpus per ID und zeigt alle Turns.
Aufruf: py lookup_conv.py
"""

import xml.etree.ElementTree as ET
import html

XML_PATH = r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PRED_PATH = r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

print("Lade Predators ...")
with open(PRED_PATH) as f:
    predators = set(l.strip() for l in f if l.strip())

print("Parse XML (einmalig, dauert ~10s) ...")
tree = ET.parse(XML_PATH)
root = tree.getroot()
conv_map = {c.get("id"): c for c in root.findall("conversation")}
print(f"  {len(conv_map)} Conversations geladen.\n")

while True:
    conv_id = input("Conv-ID eingeben (oder 'q' zum Beenden): ").strip()
    if conv_id.lower() == "q":
        break

    conv = conv_map.get(conv_id)
    if not conv:
        print(f"  NICHT GEFUNDEN: {conv_id}\n")
        continue

    authors = {m.findtext("author") for m in conv.findall("message")}
    is_grooming = bool(authors & predators)
    predator_authors = authors & predators

    print(f"\n  Conv-ID : {conv_id}")
    print(f"  Grooming: {'JA' if is_grooming else 'NEIN'}")
    if predator_authors:
        print(f"  Predator: {', '.join(predator_authors)}")
    print()

    for msg in conv.findall("message"):
        author = msg.findtext("author") or ""
        text = html.unescape(msg.findtext("text") or "").strip()
        marker = " [PREDATOR]" if author in predators else ""
        print(f"  [{msg.get('line'):>3}] {author[:12]}...{marker}")
        print(f"        {text[:120]}")
    print()
