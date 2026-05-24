"""
Extrahiert ~50 Beispiel-Turns pro Behavior-Label aus dem PAN12-Trainingskorpus.
Basiert auf keyword-basierten Heuristiken; gedacht als Annotationshilfe.
"""

import xml.etree.ElementTree as ET
import re
import html
from collections import defaultdict

XML_PATHS = [
    r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-2012-05-01.xml",
    r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-test-corpus-2012-05-17.xml",
]
PRED_PATH = r"C:\Users\mugur\Bachelorarbeit\eSPD-datasets\PAN12\raw_dataset\pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

TARGET_PER_LABEL = 50
MAX_PER_CONV = 3  # Diversität: max. Beispiele aus derselben Conversation

LABELS_ORDER = [
    "NEUTRAL",
    "VERTRAUENSAUFBAU",
    "INFORMATIONSGEWINNUNG",
    "GEHEIMHALTUNG/ISOLATION",
    "SEXUALISIERUNG",
    "KONTROLLE/NOETIGUNG",
    "OFFLINE-ESKALATION",
]

# ── Keyword-Pattern pro Label ──────────────────────────────────────────────────

PATTERNS = {
    "VERTRAUENSAUFBAU": [
        r"\byou('re| are) (so |really |very )?(nice|sweet|cool|special|amazing|pretty|cute|beautiful|kind|great|awesome|funny|interesting|mature|different|smart)\b",
        r"\bi (really |so )?(like|love|adore|enjoy) (talking to |chatting with )?you\b",
        r"\byou can trust me\b",
        r"\bi('m| am) here for you\b",
        r"\bwe have (so much |a lot )in common\b",
        r"\byou('re| are) (not like|different from) (other|most)\b",
        r"\bi understand you\b",
        r"\byou seem (really |so |very )?(mature|smart|cool|sweet|nice|kind)\b",
        r"\byou make me (smile|happy|feel)\b",
        r"\bi care about you\b",
        r"\bspecial (girl|boy|person|friend|lady)\b",
        r"\byou'?re (so )?perfect\b",
        r"\bmy (best |favorite )?(friend|girl)\b",
        r"\bi feel (so )?(comfortable|close|connected) (with )?you\b",
        r"\bwe'?re (so )?alike\b",
        r"\btell me (more |everything )?about (your)?self\b",
        r"\bwhat are your (interests|hobbies)\b",
        r"\bwhat do you (like to do|enjoy)\b",
        r"\byou'?re (really|so) (mature|smart|funny|cute|pretty|beautiful|sweet|amazing)\b",
        r"\bi('ve| have) never (met|talked to) (anyone|a girl|a guy) like you\b",
    ],
    "INFORMATIONSGEWINNUNG": [
        r"\ba[/\s]*s[/\s]*l\b",
        r"\bhow old are you\b",
        r"\bwhat'?s your age\b",
        r"\bage\?",
        r"\bwhere (are you|do you live|are you from|you from)\b",
        r"\bwhat (city|state|country|town|part of)\b",
        r"\bwhat school\b",
        r"\bwhat grade\b",
        r"\bdo you have a (boy|girl)friend\b",
        r"\bare you (single|taken|dating)\b",
        r"\bhome alone\b",
        r"\banyone (else )?(home|there)\b",
        r"\bwhere'?s (your )?(mom|dad|parents|family)\b",
        r"\bsend (me )?(a |your )?(pic|picture|photo)\b",
        r"\bdo you have (a )?(pic|picture|photo)\b",
        r"\bwhat do you look like\b",
        r"\bhow tall are you\b",
        r"\bwhat'?s your (name|real name|number|address)\b",
        r"\bdo you have (siblings|brothers|sisters)\b",
        r"\bare your parents home\b",
        r"\bare you alone\b",
        r"\byour parents (home|around|there)\b",
        r"\bpic\?",
        r"\bpicture\?",
    ],
    "GEHEIMHALTUNG/ISOLATION": [
        # explizite Geheimhaltung
        r"\bdon'?t tell\b",
        r"\bkeep (it|this|that|our) (a )?(secret|quiet|private|between)\b",
        r"\bour (little )?secret\b",
        r"\bjust between (us|you and me)\b",
        r"\bcan (you|u) keep a secret\b",
        r"\bkeep this (quiet|between|private)\b",
        r"\bnobody (else )?(needs? to|has to|should|can|will) (know|find out)\b",
        r"\bno one (else )?(needs? to|has to|can|will) know\b",
        r"\bdon'?t mention (this|it|us|me)\b",
        r"\bdon'?t tell anyone\b",
        r"\bdon'?t tell (your )?(anyone|anybody|mom|dad|parents|friends|family)\b",
        r"\bdelete (this|the|our) (message|chat|conv|history)\b",
        # Isolation von Familie / Freunden
        r"\byour (parents?|mom|dad|family) (won'?t|can'?t|don'?t|wouldn'?t) understand\b",
        r"\bthey (won'?t|wouldn'?t|don'?t) understand\b",
        r"\byour (parents?|mom|dad|family) (are|is|would be) (too |so )?(strict|controlling|annoying|old.fashioned|upset)\b",
        r"\byour (parents?|mom|dad|friends?) (don'?t|won'?t|can'?t|doesn'?t) need to know\b",
        r"\byour parents can'?t (know|find out)\b",
        r"\bdon'?t tell your (parents?|mom|dad|family|friends?)\b",
        r"\byour friends (wouldn'?t|won'?t|don'?t) understand\b",
        r"\bonly (you and )?i (understand|know|care)\b",
        r"\bonly i understand\b",
        r"\bnobody else (understands|cares about|gets) you\b",
        r"\bjust (you and )?me\b",
        r"\bjust (the two of )?us\b",
        r"\bonly us\b",
        r"\bprivate (chat|message|conversation|talk)\b",
        r"\bthis is (just )?(between us|between you and me|our secret)\b",
        r"\byou (shouldn'?t|don'?t) need (your )?(friends|them|anyone else)\b",
        r"\bthey'?re? (just )?(jealous|using you|not real friends)\b",
        # Physische / digitale Isolation
        r"\bclose (your|the) door\b",
        r"\bgo to your (room|bedroom)\b",
        r"\bis anyone (watching|looking|behind you)\b",
        r"\bcan anyone (see|hear) (you|us)\b",
        r"\bdon'?t let (your )?(parents?|mom|dad|anyone) (see|hear|know|find)\b",
        r"\bare you on a (private|shared) (computer|pc|laptop)\b",
        r"\bthis stays between\b",
        r"\byou don'?t (have to|need to) tell (them|anyone|everybody|people)\b",
        r"\byou can trust me (with|about)?\b",
        r"\bi'?m the only one (who |that )?(understands|cares|gets you)\b",
        r"\bno one (else )?has to know\b",
        r"\bit'?s (just )?(none of their business|between us|our thing)\b",
        r"\bdon'?t worry about (what )?(others?|people|your (friends?|parents?|mom|dad)) (think|say|know)\b",
        r"\byou don'?t need (to tell )?(anyone|them|your parents?|your friends?)\b",
        r"\bwe can (keep|have) (our )?(little )?(secret|thing|chat)\b",
        r"\bfor your (eyes|ears) only\b",
        r"\bjust you and i\b",
        r"\byou and me (only|alone|together)\b",
        # Isolation: "nur ich verstehe dich"
        r"\bonly (i|me) (really )?(understand|care|love|know) you\b",
        r"\bnobody else (really )?understands? you\b",
        r"\bnobody (else )?(cares about|cares for) you\b",
        r"\bi understand you better than (anyone|anybody|them|your (friends?|parents?))\b",
        r"\byour (friends?|family|parents?|mom|dad) don'?t (really |even )?(care|understand|love|know) you\b",
        r"\bthey don'?t (really |even )?(care about|understand|know) you\b",
        r"\bi'?m (the only one|all you need|here for you always)\b",
        r"\byou don'?t need (them|your friends?|your family|anyone else)\b",
        r"\bjust (the two of us|us two|you and me)\b",
        r"\bwhen (you'?re |you are )alone\b",
        r"\bwhen no ?one'?s? (home|there|around)\b",
        r"\bwhen your (parents?|mom|dad) (is|are|isn'?t|aren'?t) (home|there|around)\b",
        r"\bis your (door|room) (locked|closed|private)\b",
        r"\bdon'?t (let|allow) (your )?(mom|dad|parents?|friends?|anyone|sister|brother) (see|read|find|know)\b",
        r"\byou can'?t tell (your )?(mom|dad|parents?|friends?|anyone)\b",
        r"\bif (your )?(mom|dad|parents?) (find|finds|found|knew) out\b",
    ],
    "SEXUALISIERUNG": [
        r"\bsex\b",
        r"\bsexy\b",
        r"\bnaked\b",
        r"\bnude\b",
        r"\bhorny\b",
        r"\bturned? (you |me )?on\b",
        r"\bmasturbat\w*\b",
        r"\borgasm\b",
        r"\bpussy\b",
        r"\bcoc?k\b",
        r"\bdick\b",
        r"\bboobs?\b",
        r"\bbreast\b",
        r"\bfuck\b",
        r"\bblow ?job\b",
        r"\boral (sex)?\b",
        r"\btouch yourself\b",
        r"\bnaughty\b",
        r"\bdirty (chat|talk|pic)\b",
        r"\bkinky\b",
        r"\bvirgin\b",
        r"\bhave you (ever )?(had )?(sex|done it)\b",
        r"\bdo you (like|enjoy) sex\b",
        r"\byour (body|breasts?|ass|butt)\b",
        r"\bshow me (your|yourself)\b",
        r"\bwant to see you (naked|nude|undressed)\b",
        r"\bhave sex\b",
    ],
    "KONTROLLE/NOETIGUNG": [
        r"\byou (have|need|must|got) to\b",
        r"\byou better\b",
        r"\bor else\b",
        r"\bi'?ll tell (your|everyone|people)\b",
        r"\byou (owe|promised|said you would)\b",
        r"\bdon'?t make me\b",
        r"\bif you don'?t\b",
        r"\byou (promised|swore|agreed)\b",
        r"\bdo (it|this|that) or\b",
        r"\bi'?m (warning|serious|not kidding)\b",
        r"\bdo what i (say|tell you|ask)\b",
        r"\bno choice\b",
        r"\byou will (do|send|come)\b",
        r"\bi'?ll (show|tell|post|share) (everyone|people|your friends)\b",
        r"\byou know what (happens|will happen|i'?ll do)\b",
        r"\bdo as i say\b",
        r"\bdon'?t disobey\b",
    ],
    "OFFLINE-ESKALATION": [
        r"\bmeet (up|in person|me|sometime)\b",
        r"\bcan we meet\b",
        r"\bwanna meet\b",
        r"\bwant to meet\b",
        r"\bcome (over|visit|see me)\b",
        r"\bpick you up\b",
        r"\bgive me your (address|number|phone)\b",
        r"\byour (phone )?number\b",
        r"\bcall me\b",
        r"\bskype\b",
        r"\bwebcam\b",
        r"\bvideo (chat|call)\b",
        r"\bface to face\b",
        r"\bin person\b",
        r"\bwhere you live\b",
        r"\byour address\b",
        r"\bwe should (meet|hang out|get together)\b",
        r"\bwant to (meet|see) you\b",
        r"\bi'?d (love|like) to (meet|see) you\b",
        r"\bget together\b",
        r"\bhang out\b",
        r"\bmy (place|house|apartment|car)\b",
        r"\bcome to my\b",
        r"\bmy address\b",
        r"\bgive me your number\b",
    ],
    "NEUTRAL": [],
}


def classify(text: str) -> str:
    t = text.lower()
    for label, patterns in PATTERNS.items():
        if label == "NEUTRAL":
            continue
        for p in patterns:
            if re.search(p, t):
                return label
    return "NEUTRAL"


# ── Daten laden ────────────────────────────────────────────────────────────────

print("Lade Predator-Liste ...")
with open(PRED_PATH, encoding="utf-8") as f:
    predators = set(line.strip() for line in f if line.strip())
print(f"  {len(predators)} Predators geladen.")

all_conversations = []
for xml_path in XML_PATHS:
    print(f"Parse {xml_path.split(chr(92))[-1]} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    convs = root.findall("conversation")
    print(f"  {len(convs)} Conversations gefunden.")
    all_conversations.extend(convs)

print(f"Gesamt: {len(all_conversations)} Conversations.")

# ── Sammeln ────────────────────────────────────────────────────────────────────

examples = {lbl: [] for lbl in LABELS_ORDER}
conv_counts = {lbl: {} for lbl in LABELS_ORDER}  # conv_id -> count pro Label

for conv in all_conversations:
    conv_id = conv.get("id")
    authors_in_conv = {msg.findtext("author") for msg in conv.findall("message")}
    has_predator = bool(authors_in_conv & predators)

    for msg in conv.findall("message"):
        author = msg.findtext("author") or ""
        raw_text = msg.findtext("text") or ""
        text = html.unescape(raw_text).strip()
        if not text or len(text) < 4:
            continue

        label = classify(text)

        if label != "NEUTRAL":
            # Grooming-Labels: nur Predator-Turns in Predator-Conversations
            if author not in predators:
                continue
        else:
            # NEUTRAL: nur Turns aus Conversations OHNE Predator (echtes Neutral-Chat)
            if has_predator:
                continue

        # Diversität: max. MAX_PER_CONV Beispiele pro Conversation und Label
        count_in_conv = conv_counts[label].get(conv_id, 0)
        if count_in_conv >= MAX_PER_CONV:
            continue

        if len(examples[label]) < TARGET_PER_LABEL:
            examples[label].append({
                "conv_id": conv_id,
                "author": author,
                "text": text,
            })
            conv_counts[label][conv_id] = count_in_conv + 1

    if all(len(examples[lbl]) >= TARGET_PER_LABEL for lbl in LABELS_ORDER):
        break

# ── Ausgabe ────────────────────────────────────────────────────────────────────

for label in LABELS_ORDER:
    items = examples[label]
    print(f"\n{'='*80}")
    print(f"  LABEL: {label}  ({len(items)} Beispiele)")
    print(f"{'='*80}")
    for i, ex in enumerate(items, 1):
        print(f"  [{i:02d}] Conv-ID: {ex['conv_id']}")
        print(f"       Autor:   {ex['author']}")
        print(f"       Text:    {ex['text'][:200]}")
        print()

print("\nFertig.")
fehlend = [(lbl, len(examples[lbl])) for lbl in LABELS_ORDER if len(examples[lbl]) < TARGET_PER_LABEL]
if fehlend:
    print("Labels mit weniger als 50 Beispielen:")
    for lbl, n in fehlend:
        print(f"  {lbl}: {n} Beispiele")
else:
    print("Alle Labels haben 50 Beispiele.")
