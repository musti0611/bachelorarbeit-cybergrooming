from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

doc.styles['Normal'].font.name = 'Calibri'
doc.styles['Normal'].font.size = Pt(11)

def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x8B, 0x00, 0x00)

def h2(text):
    doc.add_heading(text, level=2)

def p(text):
    doc.add_paragraph(text)

def b(text):
    doc.add_paragraph(text, style='List Bullet')

def kasten(text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9.5)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'FFF3CD')
    para._p.get_or_add_pPr().append(shd)

def hinweis(text):
    para = doc.add_paragraph()
    run = para.add_run('💡 ' + text)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x00, 0x60, 0x00)

# Titelseite
titel = doc.add_heading('Was wir gemacht haben – Einfach erklärt', 0)
titel.alignment = WD_ALIGN_PARAGRAPH.CENTER
untertitel = doc.add_paragraph()
untertitel.alignment = WD_ALIGN_PARAGRAPH.CENTER
untertitel.add_run(
    'Technisches Setup für die Bachelorarbeit\n'
    '"Einfluss automatisch annotierter Behavior-Features\n'
    'auf die präfix-basierte Früherkennung von Cybergrooming"\n\n'
    'Mustafa Ugurluer – TU Darmstadt / Fraunhofer SIT\n'
    f'{datetime.date.today().strftime("%d. %B %Y")}'
).font.size = Pt(12)

doc.add_page_break()

# ── EINLEITUNG ────────────────────────────────────────────────────────────────

h1('Was ist das Ziel?')

p(
    'In deiner Bachelorarbeit geht es darum, eine KI zu trainieren, die erkennt, '
    'ob jemand in einem Chat ein Kind belästigt (Cybergrooming). '
    'Aber nicht erst nachdem das Gespräch vorbei ist – sondern möglichst früh, '
    'während der Chat noch läuft.'
)
p(
    'Die große Frage ist: Wird die KI besser, wenn wir ihr zusätzliche Hinweise geben? '
    'Zum Beispiel: "Diese Nachricht klingt nach Vertrauensaufbau" oder '
    '"Diese Nachricht klingt nach Sexualisierung". Diese Hinweise nennt man '
    'in der Arbeit Behavior-Labels.'
)
p(
    'Um das herauszufinden, bauen wir zwei Versionen der KI und vergleichen sie:'
)
b('Version 1 (Baseline): Die KI liest nur den rohen Chat-Text.')
b('Version 2 (with_labels): Die KI liest den Chat-Text PLUS die Behavior-Labels davor.')
p(
    'Am Ende schauen wir: Welche Version erkennt Grooming früher und genauer? '
    'Das ist der Kern deiner Bachelorarbeit.'
)

doc.add_page_break()

# ── WAS IST EINE KI HIER EIGENTLICH? ─────────────────────────────────────────

h1('Was ist diese KI überhaupt?')

p(
    'Die KI die wir benutzen heißt BERT. Das ist ein Sprachmodell – also eine KI, '
    'die Text lesen und verstehen kann. Sie wurde von Google entwickelt und hat '
    'bereits Milliarden von Texten aus dem Internet gelesen.'
)
p(
    'Man kann sich BERT wie einen sehr belesenen Praktikanten vorstellen. '
    'Er hat schon unglaublich viel Text gelesen und versteht Sprache gut. '
    'Aber er weiß noch nicht, was Cybergrooming ist. '
    'Das müssen wir ihm erst beibringen – das nennt man Training oder Fine-Tuning.'
)
p(
    'Beim Training zeigen wir BERT tausende Beispiele: '
    '"Das hier ist ein normaler Chat" und "Das hier ist ein Grooming-Chat". '
    'Nach vielen Beispielen lernt BERT den Unterschied zu erkennen.'
)

hinweis(
    'BERT ist wie ein Hund, dem man beibringt, einen bestimmten Geruch zu erkennen. '
    'Man zeigt ihm viele Beispiele, und irgendwann kann er es selbst.'
)

doc.add_page_break()

# ── SCHRITT 1 ─────────────────────────────────────────────────────────────────

h1('Schritt 1: Python installieren')

p(
    'Bevor wir die KI trainieren können, brauchen wir eine Arbeitsumgebung. '
    'Das ist so wie ein Werkzeugkasten – ohne den geht nichts.'
)
p(
    'Wir haben Miniconda installiert. Das ist ein Programm, das Python '
    '(die Programmiersprache) und alle weiteren Werkzeuge verwaltet. '
    'Der Vorteil: Alles ist sauber getrennt. Die KI-Umgebung stört keine '
    'anderen Programme auf dem Computer.'
)

h2('Was haben wir konkret gemacht?')
p('Wir haben im Terminal folgenden Befehl eingegeben:')
kasten('winget install Anaconda.Miniconda3')
p(
    'Das hat Miniconda heruntergeladen und installiert. '
    'Danach haben wir eine eigene Umgebung namens "espd" erstellt:'
)
kasten('conda create -n espd python=3.10 -y')
p(
    '"espd" steht für "Early Sexual Predator Detection". '
    'Python 3.10 ist eine bestimmte Version der Programmiersprache – '
    'wie eine bestimmte Ausgabe eines Buchs.'
)

h2('Warum war das nötig?')
p(
    'Ohne Python können wir keine KI-Programme schreiben oder ausführen. '
    'Conda sorgt dafür, dass verschiedene Projekte sich nicht gegenseitig '
    'durcheinanderbringen – jedes Projekt hat seine eigene, saubere Umgebung.'
)

doc.add_page_break()

# ── SCHRITT 2 ─────────────────────────────────────────────────────────────────

h1('Schritt 2: Die Werkzeuge installieren')

p(
    'Python allein reicht nicht. Wir brauchen spezielle Bibliotheken – '
    'das sind fertige Code-Pakete, die anderen schon geschrieben haben, '
    'damit wir nicht alles von Grund auf neu programmieren müssen.'
)

h2('Was haben wir installiert?')
b(
    'PyTorch: Das "Gehirn" für KI-Training. Stell dir vor, '
    'das ist die Fabrik, in der das Training stattfindet.'
)
b(
    'HuggingFace Transformers: Das Paket das BERT enthält. '
    'Hier ist der "Praktikant" drin, den wir weiterbilden wollen.'
)
b(
    'Pandas: Ein Werkzeug zum Lesen und Bearbeiten von Tabellen (CSV-Dateien). '
    'Wie Excel, aber für Python.'
)
b(
    'Scikit-learn: Berechnet Metriken wie Precision und Recall – '
    'also wie gut unsere KI ist.'
)

kasten(
    'pip install torch transformers pandas scikit-learn tqdm jupyter safetensors'
)

h2('Warum PyTorch und nicht etwas anderes?')
p(
    'Das originale Programm von den Forschern (Vogt et al., 2021) nutzte '
    'ein altes Google-Werkzeug namens tflite-model-maker. Dieses Werkzeug '
    'wird seit 2023 nicht mehr gepflegt und lässt sich auf Windows nicht '
    'mehr installieren. PyTorch + HuggingFace ist der moderne Standard '
    'und wird von der ganzen Forschungswelt benutzt.'
)

doc.add_page_break()

# ── SCHRITT 3 ─────────────────────────────────────────────────────────────────

h1('Schritt 3: Den Code der Forscher herunterladen')

p(
    'Die Forscher Vogt, Leser und Akbik haben 2021 ein Paper veröffentlicht '
    'in dem sie erklärt haben, wie man Cybergrooming früh erkennt. '
    'Den dazugehörigen Code haben sie öffentlich zur Verfügung gestellt.'
)
p(
    'Wir haben zwei ihrer Programme heruntergeladen:'
)
b(
    'eSPD-lab: Das eigentliche Trainings- und Evaluationsprogramm '
    '(wir nutzen es als Referenz und Inspiration).'
)
b(
    'eSPD-datasets: Skripte, die die Datensätze vorbereiten und '
    'in das richtige Format bringen.'
)
kasten(
    'git clone https://gitlab.com/early-sexual-predator-detection/eSPD-lab.git\n'
    'git clone https://gitlab.com/early-sexual-predator-detection/eSPD-datasets.git'
)
p(
    '"git clone" bedeutet: Kopiere diesen Code von GitLab auf unseren Computer. '
    'GitLab ist wie GitHub – eine Webseite wo Entwickler ihren Code speichern.'
)

doc.add_page_break()

# ── SCHRITT 4 ─────────────────────────────────────────────────────────────────

h1('Schritt 4: Den Datensatz beschaffen')

p(
    'Eine KI braucht Daten zum Lernen. In unserem Fall brauchen wir '
    'Chatverläufe – sowohl harmlose als auch Grooming-Chats.'
)
p(
    'Wir verwenden den PANC-Datensatz. Dieser besteht aus zwei Teilen:'
)
b(
    'PAN12 (harmlose Chats): Ein öffentlicher Datensatz von einer '
    'Forschungskonferenz aus dem Jahr 2012. Kostenlos verfügbar.'
)
b(
    'ChatCoder2 (Grooming-Chats): Ein Datensatz mit echten '
    'Grooming-Konversationen. Dieser muss per E-Mail angefragt werden.'
)

h2('Was haben wir mit PAN12 gemacht?')
p(
    'PAN12 haben wir von Zenodo heruntergeladen (91 MB, kostenlos). '
    'Die Daten lagen als XML-Dateien vor – das ist ein Dateiformat '
    'wie HTML, aber für strukturierte Daten.'
)
p(
    'Die Skripte der Forscher haben diese XML-Dateien dann verarbeitet '
    'und in eine CSV-Tabelle umgewandelt. Eine CSV-Datei ist wie eine '
    'Excel-Tabelle: jede Zeile ist ein Chat-Segment, mit Label (harmlos/Grooming) '
    'und dem Text.'
)

h2('Problem: Fehler beim Speichern der Datei')
p(
    'Windows benutzt ein älteres Text-Format (cp1252) als Standard. '
    'Der Code der Forscher hatte dafür keine Vorkehrung getroffen. '
    'Beim Speichern gab es deshalb einen Fehler bei Sonderzeichen.'
)
p('Wir haben in der Datei create_csv.py eine Zeile geändert:')
kasten(
    'VORHER:  open(csvPath, "w", newline="")\n'
    'NACHHER: open(csvPath, "w", newline="", encoding="utf-8")'
)
p(
    'Das sagt dem Programm explizit: "Benutze UTF-8 zum Speichern." '
    'UTF-8 ist ein universelles Text-Format das alle Sonderzeichen kennt.'
)

h2('ChatCoder2 – E-Mail wurde gesendet')
p(
    'Für den Grooming-Teil des Datensatzes haben wir eine E-Mail an '
    'April Edwards (chatcoder.com) geschickt und den Datensatz für '
    'wissenschaftliche Zwecke im Auftrag des Fraunhofer SIT angefragt. '
    'Sobald die Antwort kommt, können wir den vollständigen PANC-Datensatz '
    'zusammenbauen.'
)

doc.add_page_break()

# ── SCHRITT 5 ─────────────────────────────────────────────────────────────────

h1('Schritt 5: Das Trainingsskript schreiben (train.py)')

p(
    'Das Herzstück des Projekts. Wir haben ein Python-Programm geschrieben, '
    'das BERT trainiert. Es heißt train.py und liegt im Ordner "training".'
)

h2('Was macht dieses Programm?')
p('In einfachen Worten läuft das Training so ab:')
b('1. Lade die Chatdaten aus der CSV-Tabelle.')
b('2. Erkläre BERT: Wenn du den Text siehst, sag ob es ein Grooming-Chat ist oder nicht.')
b('3. Zeig BERT tausende Beispiele. Nach jedem Beispiel verbessert sich BERT ein bisschen.')
b('4. Nach 3 Durchläufen (Epochen) durch alle Daten: speichere das beste Modell.')

h2('Wie wird der Text für BERT aufbereitet?')
p(
    'BERT versteht keinen rohen Text. Wir müssen den Text zuerst in Zahlen umwandeln – '
    'das nennt man Tokenisierung. Jedes Wort wird zu einer Zahl. '
    '"hey whats up" wird z.B. zu [7632, 2054, 2039]. '
    'BERT kann dann mit diesen Zahlen rechnen.'
)

h2('Wie funktioniert die Behavior-Label-Variante?')
p(
    'Bei der erweiterten Variante (with_labels) fügen wir vor jede Nachricht '
    'einen Hinweis ein. Das sieht dann so aus:'
)
kasten(
    'NORMAL:      "hey whats up how old are you"\n'
    'MIT LABEL:   "[VERTRAUENSAUFBAU] hey whats up how old are you"'
)
p(
    'BERT lernt dann: Wenn vor dem Text "[VERTRAUENSAUFBAU]" steht, '
    'ist das oft ein frühes Zeichen für Grooming. Dieses Wissen kann '
    'das Modell früher zur richtigen Entscheidung bringen.'
)

h2('Problem: Modell wurde nicht gespeichert')
p(
    'Im ersten Test war der F1-Score 0.0 (weil nur 10 Grooming-Beispiele '
    'zum Trainieren). Das Programm speicherte das Modell nur wenn es '
    'besser als bisher war. Da 0.0 nicht besser als 0.0 ist (strikt größer), '
    'wurde nichts gespeichert.'
)
p('Wir haben eine Zeile in train.py geändert:')
kasten(
    'VORHER:  if metrics["f1"] > best_f1:\n'
    'NACHHER: if metrics["f1"] >= best_f1:'
)
p(
    'Jetzt wird das Modell immer mindestens einmal gespeichert – '
    'auch wenn der Score 0.0 ist. Das ist wichtig für den Smoke-Test.'
)

doc.add_page_break()

# ── SCHRITT 6 ─────────────────────────────────────────────────────────────────

h1('Schritt 6: Das Evaluationsskript schreiben (evaluate.py)')

p(
    'Nach dem Training müssen wir wissen: Wie gut ist die KI? '
    'Aber "gut" bedeutet hier etwas Besonderes – wir wollen nicht nur wissen, '
    'ob die KI Grooming erkennt, sondern auch WIE FRÜH sie es erkennt.'
)
p(
    'Dafür haben wir ein zweites Programm geschrieben: evaluate.py.'
)

h2('Wie funktioniert die Evaluation?')
p(
    'Stell dir vor, du liest einen Chat-Verlauf Nachricht für Nachricht mit. '
    'Nach jeder neuen Nachricht fragst du die KI: "Ist das Grooming?" '
    'Das nennt sich präfix-basierte Evaluation.'
)
p('Beispiel mit 5 Nachrichten:')
kasten(
    'Nachricht 1: "hey wie gehts"    \u2192 KI sagt: 3% Wahrscheinlichkeit Grooming\n'
    'Nachricht 2: "wie alt bist du"  → KI sagt: 15%\n'
    'Nachricht 3: "bist du allein"   → KI sagt: 45%\n'
    'Nachricht 4: "schick mir fotos" → KI sagt: 82%  ← ALARM!\n'
    'Nachricht 5: "..."              → bereits Alarm ausgelöst'
)
p(
    'Je früher die KI Alarm schlägt (also bei Nachricht 2 statt Nachricht 4), '
    'desto besser. Das ist die Warnlatenz.'
)

h2('Was ist der MasterClassifier?')
p(
    'Die KI sagt nach jeder Nachricht eine Wahrscheinlichkeit (z.B. 82%). '
    'Aber wann lösen wir wirklich Alarm aus? Dafür gibt es den MasterClassifier.'
)
p(
    'Er schaut sich die letzten 10 Vorhersagen an. '
    'Der "Skeptizismus" (s) sagt: wie viele der letzten 10 müssen über 50% sein, '
    'damit Alarm ausgelöst wird?'
)
b('s=1: Bei EINER positiven Vorhersage → Alarm (sehr sensitiv, viele Fehlalarme)')
b('s=5: Bei FÜNF positiven Vorhersagen → Alarm (ausgewogen)')
b('s=10: Erst bei ZEHN positiven Vorhersagen → Alarm (sehr konservativ, wenige Fehlalarme)')
p(
    'Wir testen alle 10 Werte (s=1 bis s=10) und schauen welcher der beste ist. '
    'Das ist wichtig für deine Arbeit, weil du zeigen kannst: '
    'Mit Behavior-Labels funktioniert das Modell besser bei bestimmten s-Werten.'
)

h2('Was ist F-latency?')
p(
    'F-latency ist DIE wichtigste Zahl in deiner Arbeit. '
    'Sie kombiniert zwei Dinge in einer einzigen Zahl:'
)
b('F1-Score: Wie oft liegt die KI richtig? (Genauigkeit)')
b('Speed: Wie früh warnt die KI? (Schnelligkeit)')
kasten('F-latency = F1-Score × Speed')
p(
    'Ein Modell das alles richtig erkennt (F1=1.0) aber immer erst sehr spät '
    'warnt (Speed=0.3) hat F-latency = 0.3. '
    'Ein Modell das früh warnt (Speed=0.9) aber viele Fehler macht (F1=0.4) '
    'hat F-latency = 0.36. '
    'Nur ein Modell das BEIDES gut kann, bekommt einen hohen F-latency-Wert.'
)
hinweis(
    'F-latency ist die Metrik mit der du in der Bachelorarbeit zeigst: '
    '"Mit Behavior-Labels ist mein Modell besser."'
)

h2('Problem: Windows-Pfade funktionierten nicht')
p(
    'Beim Laden des gespeicherten Modells gab es einen Fehler. '
    'HuggingFace (das Programm das BERT verwaltet) hat den Windows-Pfad '
    'wie "C:\\Users\\mugur\\..." falsch interpretiert – es dachte, '
    'das sei ein Modell-Name aus dem Internet.'
)
p(
    'Wir haben das Laden des Modells umgeschrieben. Statt dem Standard-Befehl '
    'laden wir die einzelnen Dateien direkt:'
)
kasten(
    'VORHER (funktioniert nicht auf Windows):\n'
    '  model = BertForSequenceClassification.from_pretrained("C:/Users/...")\n\n'
    'NACHHER (funktioniert):\n'
    '  config = BertConfig.from_json_file("C:/Users/.../config.json")\n'
    '  model = BertForSequenceClassification(config)\n'
    '  model.load_state_dict(load_safetensors("C:/Users/.../model.safetensors"))'
)
p(
    'Das Ergebnis ist dasselbe – das Modell wird korrekt geladen. '
    'Wir umgehen nur die fehlerhafte Validierung von HuggingFace auf Windows.'
)

doc.add_page_break()

# ── SCHRITT 7 ─────────────────────────────────────────────────────────────────

h1('Schritt 7: Smoke-Test – funktioniert alles?')

p(
    'Bevor wir stundenlang auf dem echten Datensatz trainieren, '
    'haben wir einen schnellen Test gemacht. Das nennt man Smoke-Test: '
    'Einmal kurz einschalten und schauen ob Rauch kommt.'
)
p(
    'Wir haben einen Mini-Datensatz erstellt: 50 Chat-Segmente '
    '(40 harmlos, 10 Grooming) und das Modell nur für 1 Epoche trainiert.'
)
p(
    'Ergebnis: Das Programm lief fehlerfrei durch, das Modell wurde '
    'gespeichert, und das Evaluationsskript konnte es laden und auswerten. '
    'Der F1-Score war 0.0 – das ist aber völlig normal bei 10 Beispielen. '
    'Es ging nur darum zu prüfen: Laufen die Programme ohne Fehler?'
)
hinweis(
    'Das ist wie ein Probelauf vor dem echten Rennen. '
    'Man prüft ob das Auto anspringt – nicht ob es gewinnt.'
)

doc.add_page_break()

# ── STAND DER DINGE ───────────────────────────────────────────────────────────

h1('Aktueller Stand – Was ist fertig, was fehlt noch?')

h2('Fertig ✓')
b('Python-Umgebung mit allen Werkzeugen installiert')
b('Code der Forscher heruntergeladen')
b('PAN12-Datensatz heruntergeladen und verarbeitet')
b('E-Mail für ChatCoder2 gesendet')
b('Trainingsskript (train.py) geschrieben und getestet')
b('Evaluationsskript (evaluate.py) mit F-latency geschrieben und getestet')
b('3 Bugs gefunden und behoben (Encoding, Pfad, Speichern)')

h2('Ausstehend (wartet auf ChatCoder2 und GPU)')
b('ChatCoder2 erhalten → PANC-Datensatz zusammenbauen')
b('Echtes Training auf PANC (3 Epochen, braucht GPU, ~2 Stunden)')
b('Behavior-Labels mit LLM-Annotator generieren (separater Arbeitsschritt)')
b('Beide Modelle vergleichen und Ergebnisse dokumentieren')

doc.add_page_break()

# ── WARUM MACHEN WIR ES SO? ───────────────────────────────────────────────────

h1('Warum machen wir es so und nicht anders?')

h2('Warum nicht einfach das Original-Programm der Forscher benutzen?')
p(
    'Das Original-Programm von 2021 benutzt ein Google-Werkzeug namens '
    'tflite-model-maker. Dieses Werkzeug wurde 2023 von Google eingestellt '
    '– es wird nicht mehr weiterentwickelt und lässt sich auf Windows mit '
    'modernem Python nicht mehr installieren. '
    'Wir mussten das Training daher mit modernen Werkzeugen neu implementieren.'
)

h2('Warum HuggingFace?')
p(
    'HuggingFace ist heute der Standard in der KI-Forschung für Sprachmodelle. '
    'Fast jede Universität und jedes Forschungsinstitut benutzt es. '
    'Es ist gut dokumentiert, wird aktiv gepflegt, und macht es einfach, '
    'eigene Special Tokens (wie unsere Behavior-Labels) hinzuzufügen.'
)

h2('Warum PANC und nicht einfach PAN12?')
p(
    'PAN12 enthält zwar auch Grooming-Chats, aber diese wurden von '
    'erwachsenen Freiwilligen geschrieben die sich als Minderjährige ausgaben. '
    'Das sind keine echten Grooming-Situationen. ChatCoder2 hat qualitativ '
    'bessere Daten. PANC kombiniert das Beste aus beiden Datensätzen.'
)

h2('Warum testen wir erst lokal und dann auf GPU?')
p(
    'GPU ist wie ein Sportwagen – sehr schnell, aber man fährt damit nicht '
    'zum ersten Mal auf einer unbekannten Strecke. Erst haben wir das '
    'Setup lokal auf dem normalen Computer getestet (auch wenn es langsam ist), '
    'um sicherzustellen dass alles funktioniert. Danach nutzen wir den '
    'schnellen GPU-Rechner der Uni für das echte Training.'
)

doc.add_page_break()

# ── ORDNERSTRUKTUR ────────────────────────────────────────────────────────────

h1('Wo liegen welche Dateien?')

p('Alle Dateien des Projekts liegen in:')
kasten('C:\\Users\\mugur\\bachelorarbeit\\')
p('Die wichtigsten Ordner und Dateien:')

b('eSPD-datasets/  →  Skripte und Daten der Forscher für die Datenvorbereitung')
b('eSPD-datasets/PAN12/csv/  →  Die fertigen Trainingsdaten als CSV-Tabellen')
b('eSPD-lab/  →  Original-Code der Forscher (nur als Referenz)')
b('training/train.py  →  UNSER Trainingsskript (Baseline + with_labels)')
b('training/evaluate.py  →  UNSER Evaluationsskript (F-latency)')
b('training/models/  →  Hier werden die trainierten Modelle gespeichert')
b('training/results/  →  Hier werden die Evaluationsergebnisse gespeichert')
b('data/raw/  →  Die heruntergeladenen Rohdaten (PAN12)')

doc.add_page_break()

# ── GLOSSAR ───────────────────────────────────────────────────────────────────

h1('Kleines Glossar – Was bedeutet was?')

h2('BERT')
p('Ein Sprachmodell von Google. Kann Text lesen und verstehen. Wir trainieren es für Cybergrooming-Erkennung.')

h2('Training / Fine-Tuning')
p('BERT mit eigenen Beispielen weiterlernen lassen. Wie einem Hund einen neuen Trick beibringen.')

h2('Epoche')
p('Ein kompletter Durchlauf durch alle Trainingsdaten. Wir trainieren 3 Epochen.')

h2('Tokenisierung')
p('Text in Zahlen umwandeln, damit BERT damit rechnen kann.')

h2('Special Token')
p('Ein Sonderzeichen das wir BERT beibringen. Z.B. [VERTRAUENSAUFBAU]. BERT lernt was das bedeutet.')

h2('Precision (Genauigkeit)')
p('Von allen Chats bei denen die KI Alarm geschlagen hat: wie viele waren wirklich Grooming? Hohe Precision = wenige Fehlalarme.')

h2('Recall')
p('Von allen echten Grooming-Chats: wie viele hat die KI gefunden? Hoher Recall = nichts übersehen.')

h2('F1-Score')
p('Der Mittelwert aus Precision und Recall. Eine einzige Zahl die zeigt wie gut die KI insgesamt ist.')

h2('Warnlatenz')
p('Anzahl der Nachrichten bis die KI Alarm schlägt. Weniger ist besser.')

h2('F-latency')
p('F1-Score × Speed. Die wichtigste Zahl der Arbeit. Misst ob die KI gut UND früh warnt.')

h2('MasterClassifier / Skeptizismus')
p('Regelt wie viele positive Vorhersagen nötig sind für einen Alarm. s=1 = sofort Alarm, s=10 = erst nach vielen Signalen.')

h2('PANC')
p('Der Datensatz den wir benutzen. Kombination aus PAN12 (harmlose Chats) und ChatCoder2 (Grooming-Chats).')

h2('CSV')
p('Eine einfache Tabellendatei (wie Excel). Jede Zeile ist ein Chat-Segment mit Label und Text.')

h2('Smoke-Test')
p('Ein kurzer Test nur um zu prüfen ob alles grundsätzlich funktioniert – nicht ob die Ergebnisse gut sind.')

out = 'C:/Users/mugur/bachelorarbeit/Erklaerung_Fuer_Einsteiger.docx'
doc.save(out)
print(f'Gespeichert: {out}')
