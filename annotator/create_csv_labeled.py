"""
Create the ANNOTATED BERT CSV from the labeled PAN12 datapacks.

Mirrors create_csv.py EXACTLY (same getSegments, same isGood filter, same
MESSAGE_DELIMITER) so the output is 1:1 comparable to the baseline CSV -- the
ONLY difference is that every message is prefixed with its behavior label:

    baseline : "hey whats up  u home alone?"
    labeled  : "[NEUTRAL] hey whats up  [INFORMATIONSGEWINNUNG] u home alone?"

Run from the eSPD-datasets folder:
    py create_csv_labeled.py --dataset PAN12

Output: PAN12/csv/PAN12labeled-train.csv , PAN12labeled-test.csv
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import json
import csv
from pathlib import Path
from util import getSegments, isGood, isNonemptyMsg, MESSAGE_DELIMITER

parser = argparse.ArgumentParser(description='Create a labeled csv dataset from a labeled datapack')
parser.add_argument("--dataset", dest='dataset', required=True)
parser.add_argument("--datapackID", dest='datapackID')
args = parser.parse_args()
if args.datapackID is None:
    args.datapackID = args.dataset


def labelToken(label):
    # match the special-token spelling used in training/train.py (no / or -)
    return "[" + label.replace("/", "_").replace("-", "_") + "]"


def contentToStringLabeled(content):
    """Same as util.contentToString, but prefixes each message with [LABEL]."""
    string = ""
    for ct in content:
        if not isNonemptyMsg(ct) or ct["type"] != "message":
            continue
        labels = ct.get("labels") or []
        prefix = labelToken(labels[0]) + " " if labels else ""
        if string != "":
            string += MESSAGE_DELIMITER
        string += prefix + ct["body"]
    return string


def writeCSV(file, datapack):
    writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["label", "chatName", "segment"])  # header

    n_chats = len(datapack["chats"])
    rows = 0
    for ci, (chatName, chat) in enumerate(datapack["chats"].items(), 1):
        for i, segment in enumerate(getSegments(chat)):
            if not isGood(segment, args.dataset):
                continue
            segmentName = "%s-%s" % (chatName, i) \
                if len(segment) != len(chat["content"]) else chatName
            writer.writerow([chat["className"], segmentName, contentToStringLabeled(segment)])
            rows += 1
        if ci % 20000 == 0:
            print(f"  {ci:,}/{n_chats:,} Konversationen...")
    return rows


for datasetType in ["train", "test"]:
    datapackPath = "%s/datapacks/datapack-%s-%s-labeled.json" % (
        args.dataset, args.datapackID, datasetType)

    outPath = "%s/csv/" % args.dataset
    Path(outPath).mkdir(parents=True, exist_ok=True)
    csvPath = os.path.join(outPath, "%slabeled-%s.csv" % (args.datapackID, datasetType))

    print(f"\nLade {datapackPath} ...")
    with open(datapackPath, "r", encoding='utf-8') as f:
        datapack = json.load(f)
    print(f"Schreibe {csvPath} ...")
    with open(csvPath, 'w', newline='', encoding='utf-8') as f:
        rows = writeCSV(f, datapack)
    print(f"FERTIG {datasetType}: {rows:,} Zeilen geschrieben.")

print("\nAlle labeled-CSV-Dateien geschrieben.")
