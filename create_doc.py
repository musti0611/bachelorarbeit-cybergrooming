from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

def heading1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x8B, 0x00, 0x00)
    return p

def heading2(text):
    return doc.add_heading(text, level=2)

def heading3(text):
    return doc.add_heading(text, level=3)

def para(text):
    return doc.add_paragraph(text)

def bullet(text):
    return doc.add_paragraph(text, style='List Bullet')

def code(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    shading = OxmlElement('w:shd')
    shading.set(qn('w:val'), 'clear')
    shading.set(qn('w:color'), 'auto')
    shading.set(qn('w:fill'), 'F2F2F2')
    p._p.get_or_add_pPr().append(shading)
    return p

def note(text):
    p = doc.add_paragraph()
    run = p.add_run('Hinweis: ' + text)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x88)
    return p

# ── Titelseite ───────────────────────────────────────────────────────────────

title = doc.add_heading('Technisches Setup – Dokumentation', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run(
    'Bachelorarbeit: Einfluss automatisch annotierter Behavior-Features\n'
    'auf die präfix-basierte Früherkennung von Cybergrooming'
)
run.font.size = Pt(13)
run.font.italic = True

doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run(
    f'Mustafa Ugurluer · Matrikelnummer 2699024\n'
    f'TU Darmstadt / Fraunhofer SIT\n'
    f'{datetime.date.today().strftime("%d. %B %Y")}'
)

doc.add_page_break()

# ── 1. Überblick ─────────────────────────────────────────────────────────────

heading1('1  Überblick und Zielsetzung')
para(
    'Dieses Dokument beschreibt das vollständige technische Setup für den Modellteil '
    'der Bachelorarbeit. Konkret geht es um den Vergleich zweier BERT-basierter '
    'eSPD-Modelle (Early Sexual Predator Detection):'
)
bullet('Baseline-Modell: BERT-base-uncased trainiert auf reinem Rohtext')
bullet(
    'Erweitertes Modell: identisches BERT-base-uncased, Rohtext wird durch automatisch '
    'annotierte Behavior-Labels als Special Tokens ergänzt '
    '(z. B. [VERTRAUENSAUFBAU] hey whats up)'
)
para(
    'Der Vergleich soll zeigen, ob die Behavior-Features Recall, F1-Score und '
    'Warnlatenz (F-latency) des Modells verbessern – dies ist die zentrale '
    'Forschungsfrage der Arbeit.'
)

# ── 2. Technische Entscheidungen ─────────────────────────────────────────────

heading1('2  Technische Entscheidungen und Begründungen')

heading2('2.1  Warum nicht das originale eSPD-Lab-Framework?')
para(
    'Das originale eSPD-Lab von Vogt et al. (2021) nutzt für BERT-base die Bibliothek '
    'tflite-model-maker von Google. Diese Bibliothek ist seit 2023 offiziell eingestellt '
    '(deprecated) und lässt sich auf Windows mit modernen Python-Versionen nicht mehr '
    'installieren. Sie ist eng an ältere TensorFlow-Versionen (2.x) gebunden, die '
    'ihrerseits nicht mit Python 3.10+ kompatibel sind. Das Framework nutzt für '
    'BERT-large alternativ die Flair-Bibliothek (Version 0.8, ebenfalls veraltet). '
    'Für die vorliegende Arbeit wurde daher entschieden, die Trainingspipeline mit '
    'modernen, aktiv gewarteten Bibliotheken neu zu implementieren.'
)

heading2('2.2  Warum HuggingFace Transformers + PyTorch?')
para('HuggingFace Transformers ist der aktuelle De-facto-Standard für BERT-basierte Modelle. Die Wahl begründet sich durch:')
bullet('Aktive Wartung und Kompatibilität mit Python 3.10+')
bullet('Native Unterstützung für add_special_tokens – essenziell für die Behavior-Label-Integration')
bullet('BertForSequenceClassification als fertige Klassifikationsarchitektur')
bullet('Vollständige Kontrolle über alle Hyperparameter (lr, batch_size, epochs)')
bullet('Einfache Reproduzierbarkeit: identische Bedingungen für beide Varianten garantierbar')

heading2('2.3  Warum PANC statt PAN12 allein?')
para(
    'PAN12 enthält zwar auch Predator-Konversationen, diese stammen jedoch aus '
    'Sting-Operationen (Erwachsene als Lockvögel) und gelten als rauschbehaftet '
    '(Ringenberg et al., 2021). PANC kombiniert die negativen (nicht-prädatorischen) '
    'Segmente aus PAN12 mit den positiven (Grooming-)Konversationen aus ChatCoder2. '
    'Vogt et al. (2021) haben PANC speziell für die präfix-basierte eSPD-Evaluation '
    'konzipiert. ChatCoder2 wurde per E-Mail bei April Edwards (chatcoder.com) angefragt.'
)

heading2('2.4  Warum CPU-Training lokal?')
para(
    'Das lokale CPU-Training dient ausschließlich der Verifikation der Pipeline '
    '(Smoke-Test). Es wird sichergestellt, dass alle Komponenten korrekt '
    'zusammenspielen, bevor Rechenzeit auf dem GPU-Cluster der TU Darmstadt '
    'beansprucht wird. Für das vollständige Training auf PANC wird der '
    'GPU-Zugang der Universität verwendet.'
)

# ── 3. Durchgeführte Schritte ─────────────────────────────────────────────────

heading1('3  Durchgeführte Schritte (chronologisch)')

heading2('Schritt 1: Miniconda + Python-Umgebung')
para('Miniconda wurde als Paketmanager installiert, isolierte Umgebung espd mit Python 3.10 erstellt:')
code('winget install Anaconda.Miniconda3\nconda create -n espd python=3.10 -y')

heading2('Schritt 2: Pakete installieren')
code(
    'pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n'
    'pip install transformers==4.44.2 datasets scikit-learn pandas tqdm jupyter safetensors'
)
para('Installierte Versionen: torch 2.11.0, transformers 4.44.2, scikit-learn 1.7.2, pandas 2.3.3')

heading2('Schritt 3: eSPD-Repositories klonen')
para('Beide Repositories von Vogt et al. (MIT-Lizenz) wurden als Referenz und Datenpipeline geklont:')
code(
    'git clone https://gitlab.com/early-sexual-predator-detection/eSPD-lab.git\n'
    'git clone https://gitlab.com/early-sexual-predator-detection/eSPD-datasets.git'
)

heading2('Schritt 4: PAN12-Datensatz')
para('PAN12 von Zenodo heruntergeladen (91 MB), entpackt und in das eSPD-datasets-Verzeichnis kopiert:')
code('# Quelle: https://zenodo.org/records/3713280\n# Dateien in eSPD-datasets/PAN12/raw_dataset/')
para('Dann Datapacks und CSVs erzeugt (dabei Windows-Encoding-Bug (cp1252→UTF-8) im Originalcode behoben):')
code(
    'python eSPD-datasets/PAN12/create_datapack.py\n'
    'python eSPD-datasets/create_csv.py --dataset PAN12'
)
para('Ergebnis:')
bullet('PAN12-train.csv: 66.927 Segmente (64.912 non-predator / 2.015 predator)')
bullet('PAN12-test.csv: 155.128 Segmente (151.405 non-predator / 3.723 predator)')
note(
    'Die Klassenimbalance (~97% non-predator) wird im Training durch gewichteten '
    'Loss (pos_weight = n_neg / n_pos) ausgeglichen.'
)

heading2('Schritt 5: Trainingsskript (training/train.py)')
para('Eigenständiges HuggingFace-Trainingsskript entwickelt, das beide Varianten unterstützt:')
bullet('Lädt CSV-Daten, tokenisiert mit BertTokenizerFast (max_len=512)')
bullet('Variante with_labels: 7 Behavior-Tokens als Special Tokens hinzugefügt, Embedding-Matrix vergrößert')
bullet('BertForSequenceClassification (bert-base-uncased) mit AdamW + linearem Warmup-Scheduler')
bullet('Gewichteter Cross-Entropy-Loss für Klassenimbalance')
bullet('Bestes Modell (nach F1 auf Testset) wird automatisch gespeichert')
code(
    'python training/train.py --variant baseline --dataset PANC --epochs 3\n'
    'python training/train.py --variant with_labels --dataset PANC --epochs 3'
)

heading2('Schritt 6: Smoke-Test Training')
para(
    'Mit einem Mini-Datensatz (50 Segmente, 1 Epoche) wurde verifiziert, '
    'dass die gesamte Trainingspipeline fehlerfrei durchläuft und das '
    'Modell korrekt gespeichert wird. F1=0.0 ist bei 10 positiven '
    'Beispielen vollständig erwartet.'
)

heading2('Schritt 7: Evaluationsskript (training/evaluate.py)')
para(
    'Das Evaluationsskript implementiert die eSPD-spezifische Evaluation '
    'nach Vogt et al. (2021). Es besteht aus vier Hauptkomponenten:'
)

heading3('7a) Prefix-basierte Annotation')
para(
    'Für jeden Chat im Testset wird das Modell sequenziell aufgerufen – '
    'nach jeder neuen Nachricht einmal. Das Modell "sieht" den Chat also '
    'so wie in Echtzeit. Technisch: ein Sliding Window der letzten 50 '
    'Nachrichten wird zu einem String concateniert und als BERT-Input verwendet.'
)
code(
    '# Pseudocode der Prefix-Annotation\n'
    'for i, msg in enumerate(nonempty_messages):\n'
    '    window = nonempty_messages[max(0, i+1-50) : i+1]\n'
    '    text = " ".join(m["body"] for m in window)\n'
    '    msg["prediction"] = bert_predict_predator_prob(text)'
)

heading3('7b) MasterClassifier (Skeptizismus)')
para(
    'Der MasterClassifier entscheidet wann eine Warnung ausgelöst wird. '
    'Er hält ein gleitendes Fenster der letzten 10 Vorhersagen. Eine Warnung '
    'wird ausgelöst, wenn mindestens s (Skeptizismus) der letzten 10 Vorhersagen '
    'positiv (predator) waren. Ein höheres s macht das System konservativer '
    '(weniger False Alarms, aber spätere Warnungen).'
)
code(
    '# MasterClassifier Logik\n'
    'class MasterClassifier:\n'
    '    def __init__(self, skepticism):        # s = 1..10\n'
    '        self.history = [False] * 10\n'
    '    def add_prediction(self, is_dangerous):\n'
    '        self.history = [is_dangerous] + self.history[:-1]\n'
    '        return sum(self.history) >= self.skepticism  # Alarm?'
)
note(
    'Das Skript evaluiert automatisch alle 10 Skeptizismus-Werte (s=1..10) '
    'in einem Durchlauf und speichert alle Ergebnisse.'
)

heading3('7c) Warnlatenz und Speed-Metrik')
para(
    'Die Warnlatenz ist die Anzahl der Nachrichten bis zur ersten korrekten '
    'Warnung bei einem Predator-Chat. Die Speed-Metrik transformiert die '
    'Latenzen in einen Wert zwischen 0 und 1 via einer Sigmoid-Penaltyfunktion. '
    'Speed=1 bedeutet sofortige Warnung, Speed=0.5 entspricht einer Warnung '
    'nach dem Median-Schwellenwert (90 Nachrichten nach Vogt et al.).'
)
code(
    '# Speed-Formel nach Vogt et al. (2021)\n'
    'penalty_factor = log(3) / (90 - 1)   # Kalibrierung: 90 Nachrichten → Speed=0.5\n'
    'penalty(delay) = -1 + 2 / (1 + exp(-penalty_factor * (delay - 1)))\n'
    'speed = 1 - median([penalty(d) for d in latencies])'
)

heading3('7d) F-latency')
para(
    'F-latency ist die zentrale Gesamtmetrik der Arbeit. Sie kombiniert '
    'Erkennungsgenauigkeit (F1) und Frühzeitigkeit (Speed) in einer Zahl:'
)
code('F-latency = F1 * Speed')
para(
    'Ein Modell das alle Predator-Chats erkennt (F1=1.0) aber immer erst '
    'nach 90 Nachrichten warnt, hat F-latency = 1.0 * 0.5 = 0.5. '
    'Ein Modell das früh warnt aber viele Fehler macht hat ebenfalls '
    'einen niedrigen F-latency-Wert. Nur Modelle die beides gut machen '
    'erhalten einen hohen F-latency-Wert.'
)

heading2('Schritt 8: Smoke-Test Evaluation')
para(
    'Das Evaluationsskript wurde auf einem Mini-Datapack (30 Chats: '
    '10 predator, 20 non-predator) erfolgreich getestet. Das Skript '
    'lief ~7 Minuten auf CPU durch (auf GPU wären das ~10 Sekunden). '
    'Alle 10 Skeptizismus-Werte wurden evaluiert, Ergebnisse als JSON gespeichert.'
)
note(
    'Die Vollversion auf dem kompletten PANC-Testset wird auf dem '
    'GPU-Cluster der TU Darmstadt ausgeführt.'
)

# ── 4. Bekannte Probleme und Lösungen ────────────────────────────────────────

heading1('4  Bekannte Probleme und Lösungen')

heading2('Problem 1: Windows-Encoding-Bug im eSPD-Preprocessing')
para('Fehlermeldung: UnicodeEncodeError beim Schreiben der CSV-Dateien.')
para('Ursache: Windows nutzt standardmäßig cp1252-Encoding statt UTF-8. Der Originalcode öffnete Dateien ohne explizites Encoding.')
para('Lösung: open(..., encoding="utf-8") in create_csv.py hinzugefügt.')

heading2('Problem 2: HuggingFace Hub-Validierung blockiert Windows-Pfade')
para('Fehlermeldung: HFValidationError – Repo id must be in the form repo_name.')
para(
    'Ursache: Transformers 5.x und neuere huggingface_hub-Versionen validieren '
    'jeden Pfad als potenziellen Hub-Repo-Namen. Windows-Absolutpfade mit '
    'Laufwerksbuchstaben (C:\\...) bestehen diese Validierung nicht.'
)
para('Lösung: Modell nicht via from_pretrained() laden, sondern direkt:')
bullet('Tokenizer: BertTokenizerFast(vocab_file=...) mit expliziten Dateipfaden')
bullet('Modell: BertConfig.from_json_file() + BertForSequenceClassification(config) + load_state_dict()')
bullet('Weights-Format: model.safetensors wird priorisiert, pytorch_model.bin als Fallback')

heading2('Problem 3: Modell wurde bei F1=0.0 nicht gespeichert')
para('Ursache: Speicherbedingung war f1 > best_f1 (strikt größer). Bei F1=0.0 im ersten Epoch wurde nie gespeichert.')
para('Lösung: Bedingung auf >= geändert, sodass das Modell immer mindestens einmal gespeichert wird.')

# ── 5. Projektstruktur ────────────────────────────────────────────────────────

heading1('5  Projektstruktur')
code(
    'C:/Users/mugur/bachelorarbeit/\n'
    '├── eSPD-datasets/               # Vogt et al. Preprocessing (MIT-Lizenz)\n'
    '│   ├── PAN12/\n'
    '│   │   ├── raw_dataset/         # PAN12 XML-Rohdaten (von Zenodo)\n'
    '│   │   ├── datapacks/           # JSON-Datapacks (train/test)\n'
    '│   │   └── csv/                 # Train/Test CSVs für Training\n'
    '│   └── ChatCoder2/              # (ausstehend – Datenzugang beantragt)\n'
    '├── eSPD-lab/                    # Vogt et al. Original-Framework (Referenz)\n'
    '├── data/raw/                    # Heruntergeladene Rohdaten\n'
    '└── training/\n'
    '    ├── train.py                 # Training: Baseline + with_labels\n'
    '    ├── evaluate.py              # eSPD-Evaluation: F-latency, s=1..10\n'
    '    ├── models/                  # Gespeicherte BERT-Modelle\n'
    '    └── results/                 # Evaluationsergebnisse (JSON)'
)

# ── 6. Nächste Schritte ───────────────────────────────────────────────────────

heading1('6  Nächste Schritte')
bullet('ChatCoder2 erhalten → PANC zusammenbauen (PAN12 neg. + ChatCoder2 pos.)')
bullet('GPU-Zugang der TU Darmstadt einrichten')
bullet('Baseline auf PANC trainieren (3 Epochen, ~2h auf GPU)')
bullet('LLM-Annotator entwickeln: Few-Shot-Prompting mit den 7 Behavior-Labels')
bullet('Gesamten PANC-Datensatz mit Behavior-Labels annotieren')
bullet('Erweitertes Modell (with_labels) unter identischen Bedingungen trainieren')
bullet('evaluate.py auf beiden Modellen ausführen und Ergebnisse vergleichen')
bullet('Qualitative Fehleranalyse: bei welchen Behavior-Mustern hilft das Label am meisten?')

# ── 7. Befehle für das vollständige Training ──────────────────────────────────

heading1('7  Befehle für das vollständige Training (GPU-Cluster)')
para('Sobald PANC bereit ist und GPU-Zugang besteht, werden folgende Befehle ausgeführt:')
code(
    '# PANC-Datensatz erstellen\n'
    'python eSPD-datasets/PANC/create_datapack.py\n'
    'python eSPD-datasets/create_csv.py --dataset PANC\n\n'
    '# Baseline trainieren\n'
    'python training/train.py \\\n'
    '  --variant baseline \\\n'
    '  --data_dir eSPD-datasets/PANC/csv \\\n'
    '  --dataset PANC \\\n'
    '  --epochs 3 --batch_size 16 --lr 2e-5\n\n'
    '# Erweitertes Modell trainieren\n'
    'python training/train.py \\\n'
    '  --variant with_labels \\\n'
    '  --data_dir eSPD-datasets/PANC/csv \\\n'
    '  --dataset PANC_labeled \\\n'
    '  --epochs 3 --batch_size 16 --lr 2e-5\n\n'
    '# Beide Modelle evaluieren\n'
    'python training/evaluate.py \\\n'
    '  --model_dir training/models/PANC_baseline/best_model \\\n'
    '  --datapack eSPD-datasets/PANC/datapacks/datapack-PANC-test.json \\\n'
    '  --output_dir training/results/PANC_baseline \\\n'
    '  --variant baseline\n\n'
    'python training/evaluate.py \\\n'
    '  --model_dir training/models/PANC_with_labels/best_model \\\n'
    '  --datapack eSPD-datasets/PANC/datapacks/datapack-PANC-test.json \\\n'
    '  --output_dir training/results/PANC_with_labels \\\n'
    '  --variant with_labels'
)

# ── 8. Literatur ──────────────────────────────────────────────────────────────

heading1('8  Verwendete Quellen und Repositories')
bullet('Vogt, M., Leser, U., Akbik, A. (2021). Early Detection of Sexual Predators in Chats. ACL 2021.')
bullet('eSPD-Lab: https://gitlab.com/early-sexual-predator-detection/eSPD-lab (MIT)')
bullet('eSPD-Datasets: https://gitlab.com/early-sexual-predator-detection/eSPD-datasets (MIT)')
bullet('PAN12 Dataset: https://zenodo.org/records/3713280')
bullet('ChatCoder2: https://www.chatcoder.com/data.html (Zugang beantragt)')
bullet('HuggingFace Transformers: https://huggingface.co/bert-base-uncased')
bullet('Ringenberg et al. (2021). Implications of using Internet sting corpora. ACL-IJCNLP.')

out = 'C:/Users/mugur/bachelorarbeit/Technisches_Setup_Dokumentation.docx'
doc.save(out)
print(f'Gespeichert: {out}')
