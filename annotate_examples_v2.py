"""
Extrahiert ~50 Beispiel-Turns pro Behavior-Label aus dem PAN12-Trainingskorpus.
ALLE Beispiele stammen von Predators (auch NEUTRAL = Predator-Turns ohne Grooming-Signal).
v2: überarbeitete Patterns für KONTROLLE/NÖTIGUNG und GEHEIMHALTUNG/ISOLATION.
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
OUT_PATH = r"C:\Users\mugur\Bachelorarbeit\annotate_results_v2.txt"

TARGET_PER_LABEL = 50
MAX_PER_CONV = 3          # max Beispiele pro Conversation und Label
NEUTRAL_MIN_LEN = 25      # NEUTRAL-Turns müssen mindestens so lang sein

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
        r"\byou make me (smile|happy|feel good)\b",
        r"\bi care about you\b",
        r"\bspecial (girl|boy|person|friend|lady)\b",
        r"\byou'?re (so )?perfect\b",
        r"\bi feel (so )?(comfortable|close|connected) (with )?you\b",
        r"\bwe'?re (so )?alike\b",
        r"\byou'?re (really|so) (mature|smart|funny|cute|pretty|beautiful|sweet|amazing)\b",
        r"\bi('ve| have) never (met|talked to) (anyone|a girl|a guy) like you\b",
        r"\byou are (so |very |really )?(special|unique|different|wonderful|lovely)\b",
        r"\bi(('ve)| have) never felt (this way|like this) (about anyone|before)\b",
        r"\byou mean (so much|a lot|everything) to me\b",
        r"\byou('re| are) (so )?beautiful\b",
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
        r"\bsend (me )?(a |your )?(pic|picture|photo) of you\b",
        r"\bdo you have (a )?(pic|picture|photo) of you\b",
        r"\bwhat do you look like\b",
        r"\bhow tall are you\b",
        r"\bwhat'?s your (real name|number|address)\b",
        r"\bdo you have (siblings|brothers|sisters)\b",
        r"\bare your parents home\b",
        r"\bare you (alone|by yourself)\b",
        r"\byour parents (home|around|there)\b",
    ],
    "GEHEIMHALTUNG/ISOLATION": [
        # Explizite Geheimhaltung
        r"\bdon'?t tell (anyone|anybody|your (mom|dad|parents?|friends?|family|sister|brother))\b",
        r"\bkeep (it|this|that|our) (a )?(secret|quiet|private|between us)\b",
        r"\b(our|a) (little )?secret\b",
        r"\bjust between (us|you and me)\b",
        r"\bcan (you|u) keep a secret\b",
        r"\bnobody (else )?(needs? to|has to|should|can|will) (know|find out)\b",
        r"\bno one (else )?(needs? to|has to|can|will) know\b",
        r"\bdelete (this|the|our) (message|chat|conv|conversation|history)\b",
        r"\bthis (is |stays )?(just )?(between us|between you and me|our secret)\b",
        r"\bfor your (eyes|ears) only\b",
        r"\bwe can (keep|have) (our )?(little )?(secret|thing)\b",
        r"\byou can'?t tell (your )?(mom|dad|parents?|friends?|anyone)\b",
        r"\bdon'?t say (a word|anything) (to |about )?(anyone|anybody|them|your (mom|dad|parents?))\b",
        r"\bkeep (this |it )?(to yourself|quiet|private|between us)\b",
        r"\bdon'?t (mention|say|share) (this|it|us|anything) to (anyone|anybody|them)\b",
        r"\bi won'?t tell if you won'?t\b",
        r"\bssh+\b",
        r"\bshh+\b",
        r"\bmum'?s? the word\b",
        r"\bthis stays (here|with us|between us|private)\b",
        r"\bprivate (between|just for|only for) (us|you and me)\b",
        # Isolation: Familie/Freunde verstehen nicht
        r"\byour (parents?|mom|dad|family) (won'?t|can'?t|don'?t|wouldn'?t) understand\b",
        r"\bthey (won'?t|wouldn'?t|don'?t) understand (us|you|this|me)\b",
        r"\byour (parents?|mom|dad|family|friends?) (don'?t|won'?t|can'?t|doesn'?t) need to know\b",
        r"\bdon'?t tell your (parents?|mom|dad|family|friends?)\b",
        r"\bif (your )?(mom|dad|parents?) (find|finds|found|knew) out\b",
        r"\bthey don'?t (deserve|need) to know\b",
        r"\bforget (about )?(them|your friends?|those (people|friends))\b",
        r"\byou('?re| are) better off without (them|your (friends?|family))\b",
        r"\bthey('?re| are) (not good|bad) for you\b",
        r"\bthey don'?t deserve you\b",
        r"\bstay away from (them|those people|your (friends?|ex))\b",
        r"\bwhy do you (hang out|talk|spend time) with (them|him|her)\b",
        r"\byou don'?t need (him|her|them) (anymore|in your life)\b",
        r"\bi'?m (better|more important) (for you|than (they|he|she|them) (are|is))\b",
        # Isolation: nur ich verstehe/liebe dich
        r"\bonly (you and )?i (really )?(understand|know|care|love) you\b",
        r"\bi understand you better than (anyone|anybody|them|your (friends?|parents?))\b",
        r"\byour (friends?|family|parents?|mom|dad) don'?t (really |even )?(care|understand|love|know) you\b",
        r"\bnobody else (really )?understands? you\b",
        r"\bnobody (else )?(cares about|cares for) you\b",
        r"\byou don'?t need (them|your friends?|your family|anyone else)\b",
        r"\bi'?m the only one (who |that )?(understands|cares|gets you)\b",
        # Physische/digitale Isolation
        r"\bclose (your|the) door\b",
        r"\bdon'?t let (your )?(parents?|mom|dad|anyone|sister|brother) (see|read|find|know|hear)\b",
        r"\bcan anyone (see|hear) (you|us)\b",
        r"\bis anyone (watching|looking|behind you)\b",
        r"\bwhen (your )?(parents?|mom|dad) (is|are|isn'?t|aren'?t) (home|there|around)\b",
        r"\bwhen no ?one'?s? (home|there|around)\b",
        r"\bgo somewhere (private|alone|quiet)\b",
        r"\bfind (a |somewhere |some )(private|quiet|alone) (place|spot|room)?\b",
        r"\bare you (somewhere |in a )?(private|alone|by yourself)\b",
        r"\bwait (until|till|til) (they('re| are)|your (mom|dad|parents?) (is|are)) (gone|asleep|out|away|not home)\b",
        r"\bwhen (they('re| are)|your (mom|dad|parents?) (is|are)) (asleep|gone|out|away|not (home|around|there))\b",
        r"\bis (your |the )?door (locked|closed)\b",
        r"\bgo to your (room|bedroom) (and |so )?(close|lock|shut) the door\b",
        r"\bno one (can|will) (see|hear|know|find out)\b",
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
        r"\bshow me (your body|yourself naked|yourself nude)\b",
        r"\bwant to see you (naked|nude|undressed)\b",
        r"\bhave sex with\b",
        r"\bmake love\b",
        r"\bfinger\w* (yourself|you)\b",
        r"\bcum\b",
    ],
    "KONTROLLE/NOETIGUNG": [
        # Explizite Drohungen
        r"\bi'?ll (tell|show|send|post|share|forward) (everyone|people|all your friends|your parents?|your mom|your dad|your school)\b",
        r"\bor (else|i will|i'll|i'm going to) (tell|show|post|share|hurt|ruin)\b",
        r"\bif you don'?t.{0,50}(i will|i'll|i'm going to)\b",
        r"\bi'?ll (expose|report|ruin|destroy) you\b",
        r"\bi'?ll (post|share|send) (your |the )?(pic|picture|photo|video|nudes?)\b",
        # Erpressung / Schuld
        r"\byou (owe|promised|swore) me\b",
        r"\byou said you (would|were going to) (do|send|come|meet|show)\b",
        r"\bafter (everything|all) (i'?ve |i )(done|given|shared) (for|with) you\b",
        r"\byou (made|started) (a |this )?(deal|promise|agreement) with me\b",
        # Ultimaten
        r"\byou have no (choice|option)\b",
        r"\bno choice\b",
        r"\byou better (do|send|come|show|give|not)\b",
        r"\bdon'?t make me\b",
        r"\bdo (what|as) i (say|tell|ask)\b",
        r"\bi'?m (warning|serious) (you)?\b",
        r"\bi'?m not (joking|kidding|playing) (around)?\b",
        r"\bdon'?t (ignore|disobey|test|lie to) me\b",
        r"\byou'?ll regret (it|this|that)\b",
        r"\bdo (it|this|that) (now|right now|or)\b",
        r"\byou will (send|do|come|meet|show|give) (it|me|this|that)\b",
        r"\bi'?ll make (sure|you (regret|pay))\b",
        r"\byou (know|see) what (happens|will happen) (when|if) you\b",
        r"\bdon'?t make me (angry|upset|mad)\b",
        r"\bi (warned|told) you\b",
    ],
    "OFFLINE-ESKALATION": [
        r"\bmeet (up|in person|me|sometime|soon)\b",
        r"\bcan we meet\b",
        r"\bwanna meet\b",
        r"\bwant to meet\b",
        r"\bcome (over|visit|see me)\b",
        r"\bpick you up\b",
        r"\bgive me your (address|phone( number)?)\b",
        r"\byour (phone )?number\b",
        r"\bcall me (sometime|later|tonight|when)\b",
        r"\bskype (me|with me|together)\b",
        r"\bwebcam\b",
        r"\bvideo (chat|call) (with me|together|sometime)\b",
        r"\bface to face\b",
        r"\bin person\b",
        r"\bwhere (exactly )?you live\b",
        r"\byour (home )?address\b",
        r"\bwe should (meet|hang out|get together)\b",
        r"\b(want|love|like) to (meet|see) you (in person|sometime|soon)\b",
        r"\bi'?d (love|like) to (meet|see) you\b",
        r"\bget together (sometime|soon|with me)\b",
        r"\bhang out (with me|sometime|soon)\b",
        r"\bmy (place|house|apartment)\b",
        r"\bcome to my\b",
    ],
    "NEUTRAL": [],
}


def classify(text: str) -> str:
    t = text.lower()
    for label in LABELS_ORDER:
        if label == "NEUTRAL":
            continue
        for p in PATTERNS[label]:
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
    name = xml_path.split("\\")[-1]
    print(f"Parse {name} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    convs = root.findall("conversation")
    print(f"  {len(convs)} Conversations gefunden.")
    all_conversations.extend(convs)

print(f"Gesamt: {len(all_conversations)} Conversations.\n")

# ── Sammeln ────────────────────────────────────────────────────────────────────

examples = {lbl: [] for lbl in LABELS_ORDER}
conv_counts = {lbl: {} for lbl in LABELS_ORDER}

for conv in all_conversations:
    conv_id = conv.get("id")
    authors_in_conv = {msg.findtext("author") for msg in conv.findall("message")}
    has_predator = bool(authors_in_conv & predators)

    if not has_predator:
        continue  # Alle Labels: nur Predator-Conversations

    for msg in conv.findall("message"):
        author = msg.findtext("author") or ""
        if author not in predators:
            continue  # Nur Predator-Turns

        raw_text = msg.findtext("text") or ""
        text = html.unescape(raw_text).strip()
        if not text or len(text) < 4:
            continue

        label = classify(text)

        # NEUTRAL: Mindestlänge erzwingen, damit "lol" etc. rausfallen
        if label == "NEUTRAL" and len(text) < NEUTRAL_MIN_LEN:
            continue

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

lines = []
for label in LABELS_ORDER:
    items = examples[label]
    lines.append("=" * 80)
    lines.append(f"  LABEL: {label}  ({len(items)} Beispiele)")
    lines.append("=" * 80)
    for i, ex in enumerate(items, 1):
        lines.append(f"  [{i:02d}] Conv-ID: {ex['conv_id']}")
        lines.append(f"       Autor:   {ex['author']}")
        lines.append(f"       Text:    {ex['text'][:300]}")
        lines.append("")
    lines.append("")

output = "\n".join(lines)
print(output)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\nErgebnisse gespeichert: {OUT_PATH}")

fehlend = [(lbl, len(examples[lbl])) for lbl in LABELS_ORDER if len(examples[lbl]) < TARGET_PER_LABEL]
if fehlend:
    print("\nLabels mit weniger als 50 Beispielen:")
    for lbl, n in fehlend:
        print(f"  {lbl}: {n} Beispiele")
else:
    print("\nAlle Labels haben 50 Beispiele.")
