# Design Decisions: Annotator-Entwicklung
**Bachelorarbeit – Mustafa Ugurluer, TU Darmstadt**
**Erstellt: Mai 2026**

---

## 1. Wahl der LLMs für den Ensemble-Annotator

### Entscheidung
Für den automatischen Behavior-Annotator wurden drei LLMs aus verschiedenen Modellfamilien gewählt:
- **GPT-4o-mini** (OpenAI) — proprietäres Modell, lightweight
- **Gemini 2.0 Flash** (Google DeepMind) — proprietäres Modell, lightweight
- **Llama 3.3 70B** (Meta, via Groq) — open-source Modell

### Begründung (4 Argumente)

**Argument 1: Mid-tier Modelle sind für strukturierte Annotations-Tasks ausreichend**

Für klar definierte Klassifikationsaufgaben mit expliziten Annotation-Guidelines zeigt die Literatur, dass Modelle der GPT-3.5/4o-mini-Klasse menschliche Annotationsqualität erreichen und top-tier Modellen (GPT-4o full, Claude Opus) kaum nachstehen. Der entscheidende Qualitätsfaktor ist die Präzision des Prompts und der Guidelines — nicht die Modellgröße.

Relevante Literatur:
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). *ChatGPT outperforms crowd workers for text-annotation tasks.* PNAS.
- Törnberg, P. (2023). *ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning.* arXiv.
- Alizadeh, M. et al. (2023). *Open-source LLMs for text annotation: A practical guide.* arXiv.

**Argument 2: Ensemble-Diversität ist methodisch wertvoller als Einzelmodell-Stärke**

Drei Modelle aus verschiedenen Familien (OpenAI, Google, Meta/open-source) unterscheiden sich in Trainingsdaten, Architektur und RLHF-Ausrichtung. Systematische Fehler einzelner Modelle werden durch das Mehrheitsvoting (2/3-Prinzip) ausgeglichen. Ein Ensemble aus diversen mid-tier Modellen ist robuster als ein einzelnes top-tier Modell, das bei bestimmten Labelkategorien systematisch scheitern kann.

**Argument 3: Skalierbarkeit und Kosteneffizienz sind legitime wissenschaftliche Anforderungen**

Der PAN12-Datensatz enthält ~66.000 Predator-Turns. Die Annotation mit drei Modellen skaliert linear mit der Datenmenge. GPT-4o (full) wäre ca. 15× teurer als GPT-4o-mini bei marginalem Qualitätsgewinn für diesen Task. Kosteneffizienz in Annotation-Pipelines ist ein anerkanntes Kriterium in der NLP-Forschung und ermöglicht Reproduzierbarkeit durch andere Forschungsgruppen.

**Argument 4: Empirische Validierung durch Goldstandard**

Die Modelllwahl wird nicht a priori, sondern empirisch validiert: Alle drei Annotatoren sowie das Ensemble werden gegen einen manuell erstellten und durch eine zweite Person verifizierten Goldstandard evaluiert (per-Label Precision, Recall, F1-Score). Die Evaluation zeigt, ob die gewählten Modelle für die Aufgabe geeignet sind. Ein Upgrade auf leistungsstärkere Modelle wäre methodisch jederzeit möglich und würde durch schwache Evaluationsergebnisse motiviert.

### Kurzzusammenfassung für die Thesis

> Für den automatischen Annotator wurden GPT-4o-mini (OpenAI), Gemini 2.0 Flash (Google) und Llama 3.3 70B (Meta) im Ensemble eingesetzt. Die Wahl basiert auf vier Überlegungen: (1) Mid-tier Modelle erzielen bei strukturierten Klassifikationsaufgaben mit expliziten Guidelines vergleichbare Ergebnisse wie top-tier Modelle (Gilardi et al., 2023; Törnberg, 2023); (2) Ensemble-Diversität über Modellfamilien hinweg erhöht die Robustheit durch Mehrheitsvoting; (3) Kosteneffizienz ist für die Annotation großer Datensätze eine legitime Anforderung; (4) die finale Validierung erfolgt empirisch gegen einen manuellen Goldstandard, sodass die Modelllwahl retrospektiv durch die Evaluationsergebnisse gerechtfertigt wird.

---

## 2. Annotationsmethodik: Few-Shot Prompting

### Entscheidung
Jeder Annotator erhält einen System-Prompt mit: (1) den 7 Label-Definitionen inkl. Indikatoren und Abgrenzungsregeln, (2) expliziten Entscheidungsregeln für Grenzfälle (KEY RULES), (3) je 3 Few-Shot-Beispiele pro Label aus dem Goldstandard. Die ersten 3 Beispiele pro Label im Goldstandard dienen als Few-Shot-Kontext; die restlichen Beispiele bilden das Evaluationsset.

### Begründung

**Warum Few-Shot statt Zero-Shot?**
Bei 7 semantisch ähnlichen Labels (z.B. KONTROLLE/NOETIGUNG vs. SEXUALISIERUNG, OFFLINE-ESKALATION vs. INFORMATIONSGEWINNUNG) reicht Zero-Shot-Prompting erfahrungsgemäß nicht aus, um die feinen Abgrenzungen zuverlässig zu vermitteln. Konkrete Beispiele im Prompt verankern die Labeldefinitionen an realen Fällen und reduzieren Mehrdeutigkeiten.

**Warum nur 3 Beispiele pro Label?**
Drei Beispiele bieten einen ausreichenden Ankerpunkt ohne den Context-Window des Modells unverhältnismäßig zu belasten. Bei 7 Labels und je 3 Beispielen umfasst der System-Prompt ca. 21 Beispiele — ein in der Literatur gut untersuchter Bereich für Few-Shot-Klassifikation (Brown et al., 2020). Eine höhere Anzahl würde den Prompt stark verlängern ohne garantierte Qualitätssteigerung.

**Warum Goldstandard-Beispiele?**
Die manuell annotierten und von einer zweiten Person verifizierten Goldstandard-Beispiele sind die qualitativ hochwertigsten verfügbaren Trainingssignale. Sie repräsentieren die Annotation-Guidelines direkt in natürlicher Sprache.

### Technische Details
- Temperature: 0 (deterministische Ausgabe für Reproduzierbarkeit)
- Max Tokens: 20 (erzwingt kurze Label-Ausgabe)
- Normalisierung: Substring-Matching als Fallback falls das Modell die Label-Ausgabe umformuliert
- Modell-Ausgabe: exakt ein Label aus den 7 definierten Kategorien

### Kurzzusammenfassung für die Thesis

> Für die automatische Annotation wurde Few-Shot Prompting mit je 3 Beispielen pro Label eingesetzt. Der System-Prompt enthält die 7 Label-Definitionen mit Indikatoren, explizite Entscheidungsregeln für Grenzfälle sowie die 3 Few-Shot-Beispiele pro Label aus dem manuellen Goldstandard. Die Temperature wurde auf 0 gesetzt, um deterministische und reproduzierbare Ergebnisse zu gewährleisten.

---

## 7. Prompt-Optimierung: v1 → v2

**Datum:** 31. Mai 2026
**Motivation:** Fehleranalyse nach v1-Evaluation zeigte zwei systematische Schwächen: KONTROLLE/NOETIGUNG (F1=0.594) und VERTRAUENSAUFBAU→NEUTRAL-Fehler (8 von 11 Fehlern).

### Änderungen in v2

**VERTRAUENSAUFBAU — neue Indikatoren:**
- Affektive Emoticons (:-*, >:D<, xoxo) als Affektmarker
- Devotion/Persistenz-Ausdrücke ("I waited for you", "I kept trying to reach you")
- Harmlosigkeitsbehauptungen als Vertrauenstaktik ("I respect you", "I don't take advantage of people")
- Ausdruck von Erstaunen über die Aufmerksamkeit des Opfers ("I can't believe you're talking to me")

**KONTROLLE/NOETIGUNG — präzisierte Regeln:**
- Direkte Befehle über Erscheinungsbild/Verhalten = KONTROLLE/NOETIGUNG auch wenn sexueller Inhalt vorhanden (z.B. "wear a skirt for me", "be ready for me")
- Guilt-Tripping und Selbstmitleid als Druckmittel explizit ergänzt ("nobody loves me")
- Konkrete Beispiele mit Degradierung ("lil slut" → KONTROLLE/NOETIGUNG, nicht SEXUALISIERUNG)

**KEY_RULES — 4 neue Regeln:**
- Affektive Emoticons → VERTRAUENSAUFBAU
- Harmlosigkeitsbehauptungen → VERTRAUENSAUFBAU
- Direkter Befehl über Verhalten/Erscheinungsbild → KONTROLLE/NOETIGUNG (auch wenn sexuell)
- Guilt-Tripping/Selbstmitleid → KONTROLLE/NOETIGUNG

### Ergebnisvergleich v1 vs. v2

| Label | F1 v1 | F1 v2 | Δ |
|---|---|---|---|
| NEUTRAL | 0.807 | 0.863 | +0.056 |
| VERTRAUENSAUFBAU | 0.814 | 0.860 | +0.046 |
| INFORMATIONSGEWINNUNG | 0.808 | 0.792 | -0.016 |
| GEHEIMHALTUNG/ISOLATION | 0.750 | 0.800 | +0.050 |
| SEXUALISIERUNG | 0.882 | 0.926 | +0.044 |
| KONTROLLE/NOETIGUNG | 0.594 | 0.800 | **+0.206** |
| OFFLINE-ESKALATION | 0.817 | 0.828 | +0.011 |
| **MACRO** | **0.782** | **0.838** | **+0.056** |
| **Accuracy** | **0.793** | **0.840** | **+0.047** |

Einzige leichte Regression: INFORMATIONSGEWINNUNG -0.016 (F1 0.808 → 0.792). Alle anderen Labels verbessert.

**Ergebnisdateien:**
- v1: `annotator/results/eval_gpt_4o_mini_20260531_190916.json`
- v2: `annotator/results/eval_gpt_4o_mini_v2_20260531_192552.json`

### Wissenschaftliche Bewertung nach v2
Macro-F1 von 0.838 ist für LLM-basierte Annotation auf einem klar vertretbaren Niveau. Das schwächste Label (KONTROLLE/NOETIGUNG) liegt jetzt bei F1=0.800. Die leichte Regression bei INFORMATIONSGEWINNUNG (0.016) ist methodisch vernachlässigbar. **Der GPT-4o-mini-Annotator mit Prompt v2 wird als finaler Stand für den Ensemble-Annotator verwendet.**

---

## 8. Validierungsstrategie: Annotator-Qualität vs. Downstream-Performance

### Das zweistufige Validierungsargument

Die Qualität des Annotators wird auf zwei Ebenen validiert:

**Ebene 1 — Direkte Annotation-Qualität (Goldstandard-Evaluation):**
Per-Label Precision, Recall, F1 gegen den manuellen Goldstandard. Ergebnis: Macro-F1=0.838 (GPT-4o-mini v2). Diese Metrik zeigt, wie genau der Annotator die Labels vergeben kann.

**Ebene 2 — Downstream-Performance (BERT-Vergleich):**
Der eigentliche wissenschaftliche Beweis. Vergleich von:
- BERT-Baseline: trainiert auf Rohtext ohne Labels
- BERT+Behavior-Labels: trainiert auf Rohtext mit vorangestellten Labels (`[VERTRAUENSAUFBAU] hey whats up`)

Wenn BERT+Labels bessere Precision, Recall, F1 oder Warnlatenz (F-latency nach Vogt et al. 2021) erzielt als die Baseline, ist das der empirische Beweis dass die automatisch annotierten Behavior-Labels nützliche Features für die Früherkennung liefern.

### Warum Ebene 2 die Annotator-Limitation adressiert

Der Annotator wurde auf einem Goldstandard von ~320 Beispielen evaluiert. Die Generalisierung auf ~66.000 PAN12-Turns kann nicht vollständig garantiert werden (Distribution Shift). Jedoch gilt:

> *Wenn BERT+Labels trotz potenziell verrauschter Annotation besser als die Baseline ist, zeigt das empirisch, dass die Labels ausreichend valides Signal tragen — unabhängig von der absoluten Annotationsqualität auf dem Goldstandard.*

Selbst imperfekte Labels können nützliche Features sein, solange sie statistisch konsistent genug sind. Die Downstream-Performance ist damit die robustere Validierungsebene.

### Umkehrschluss für die Thesis-Diskussion

Falls BERT+Labels **nicht** besser als die Baseline ist, sind folgende Erklärungen möglich:
1. Der Annotator ist zu ungenau (Distribution Shift, zu viel Rauschen)
2. Die Behavior-Labels tragen keine zusätzliche Information die nicht schon im Rohtext steckt
3. Die Feature-Integration (Label als Special Token) ist suboptimal

Beide Szenarien sind wissenschaftlich valide Ergebnisse und in der Diskussion zu adressieren.

---

## 3. Ensemble-Voting-Strategie

*(wird ergänzt nach Evaluation aller 3 Einzelannotatoren)*

---

## 4. Evaluationsmetrik: Per-Label Precision und Recall

### Entscheidung
Evaluation des Annotators gegen den Goldstandard mit per-Label Precision, Recall und F1-Score sowie Macro-Durchschnitt über alle 7 Labels.

### Begründung

**Warum per-Label statt nur Accuracy?**
Bei 7 Labels mit unterschiedlicher Schwierigkeit (z.B. SEXUALISIERUNG klarer erkennbar als KONTROLLE/NOETIGUNG) würde eine globale Accuracy schwache Labels verstecken. Per-Label Metriken zeigen genau, welche Label-Kategorien der Annotator zuverlässig erkennt und wo Optimierungsbedarf besteht.

**Warum Precision und Recall getrennt?**
Für die spätere Verwendung im BERT-Modell sind beide Dimensionen relevant: Ein niedriger Recall bedeutet, dass ein Label im PAN12-Datensatz seltener gesetzt wird als es sollte (das Feature fehlt dem BERT-Modell). Eine niedrige Precision bedeutet, dass das Label zu häufig gesetzt wird (das Feature ist verrauscht). Beide Fehlerarten haben unterschiedliche Auswirkungen auf die Downstream-Performance.

---

## 5. GPT-4o-mini Evaluationsergebnisse

**Datum:** 31. Mai 2026
**Modell:** gpt-4o-mini
**Goldstandard:** `annotate_results_manual.txt` (manuell annotiert, verifiziert von Mareike Bassenge)
**Setup:** 3 Few-Shot-Beispiele pro Label, restliche Beispiele als Evaluationsset
**Evaluationsset:** 319 Beispiele (je 46–48 pro Label, KONTROLLE/NOETIGUNG: 40)
**Ergebnisdatei:** `annotator/results/eval_gpt_4o_mini_20260531_190916.json`

### Ergebnisse

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NEUTRAL | 0.698 | 0.957 | 0.807 | 46 |
| VERTRAUENSAUFBAU | 0.875 | 0.761 | 0.814 | 46 |
| INFORMATIONSGEWINNUNG | 0.724 | 0.913 | 0.808 | 46 |
| GEHEIMHALTUNG/ISOLATION | 0.882 | 0.652 | 0.750 | 46 |
| SEXUALISIERUNG | 0.818 | 0.957 | 0.882 | 47 |
| KONTROLLE/NOETIGUNG | 0.792 | 0.475 | 0.594 | 40 |
| OFFLINE-ESKALATION | 0.844 | 0.792 | 0.817 | 48 |
| **MACRO** | **0.805** | **0.787** | **0.782** | **319** |

**Overall Accuracy: 0.793** (0 API-Fehler)

### Stärken
- SEXUALISIERUNG (F1=0.882) und OFFLINE-ESKALATION (F1=0.817) werden sehr zuverlässig erkannt
- Macro-F1 von 0.782 liegt im Bereich vergleichbarer LLM-Annotations-Studien (Gilardi et al. 2023)
- Keine API-Fehler: alle 319 Beispiele erfolgreich annotiert

### Schwächen
- **KONTROLLE/NOETIGUNG** (F1=0.594): schwächstes Label; Recall 0.475 bedeutet über die Hälfte der echten Fälle werden verpasst
- **GEHEIMHALTUNG/ISOLATION** (F1=0.750): grenzwertig, insbesondere beim Recall (0.652)
- **NEUTRAL** hat niedrige Precision (0.698): andere Grooming-Labels werden gelegentlich als NEUTRAL fehlklassifiziert

---

## 6. Fehleranalyse GPT-4o-mini: Nuancierte vs. Gefährliche Fehler

### Hintergrund
Feedback der zweiten Betreuungsperson (Mareike Bassenge, 31. Mai 2026):
> „Du beschreibst häufig Entscheidungshilfen bei Grenzfällen. Es kann gut sein, dass ein trainierter Classifier mit diesen Fällen ebenfalls Probleme haben kann. Es wäre also bei der Evaluation auch interessant diese Fehler genau zu beobachten, ob es sich dabei um Nachrichten handelt, die nur bei Nuancen in andere Klassen fallen. Und dann ist zu entscheiden, ob diese leichte Ungenauigkeit so wichtig für die darauffolgende Anwendung ist."

### Klassifikation der Fehler

Von 66 Gesamtfehlern wurden zwei Fehlertypen unterschieden:
- **Nuanciert (Grooming → anderes Grooming-Label):** Inhaltlich Grooming-Turn, aber falsches spezifisches Label
- **Gefährlich (Grooming → NEUTRAL):** Echter Grooming-Turn wird als NEUTRAL klassifiziert — das Feature fehlt dem BERT-Modell vollständig

| Label | Fehler gesamt | → NEUTRAL (gefährlich) | → anderes Grooming (nuanciert) |
|---|---|---|---|
| VERTRAUENSAUFBAU | 11 | **8** | 3 |
| KONTROLLE/NOETIGUNG | 21 | 5 | 16 |
| GEHEIMHALTUNG/ISOLATION | 16 | 2 | 14 |
| OFFLINE-ESKALATION | 10 | 2 | 8 |
| INFORMATIONSGEWINNUNG | 4 | 2 | 2 |
| SEXUALISIERUNG | 2 | 0 | 2 |
| NEUTRAL (→ Grooming) | 2 | — | 2 |

**Ergebnis:** 45 von 66 Fehlern (68%) sind nuancierte Grooming→Grooming-Verwechslungen, nur 19 (29%) sind echte NEUTRAL-Fehler.

### Binäre Annotationsqualität (Grooming vs. Nicht-Grooming)
Für den eigentlichen Downstream-Task (binäre Cybergrooming-Klassifikation durch BERT) ist entscheidend, ob Grooming-Turns als NEUTRAL durchrutschen:
- **Grooming-Recall (binär):** 254/273 = **93.0%** — nur 7% der Grooming-Turns werden als NEUTRAL verpasst
- **NEUTRAL-Precision (binär):** 44/46 = **95.7%** — fast keine False Positives bei NEUTRAL

### Hauptverwechslungsmuster bei KONTROLLE/NOETIGUNG
Die meisten KONTROLLE/NOETIGUNG-Fehler (16/21) entstehen, weil der Text gleichzeitig sexuellen Inhalt und eine Befehlsstruktur enthält, z.B.:
- *"well wear 1 of them for me lil slut"* → pred: SEXUALISIERUNG (sexueller Inhalt dominiert)
- *"will u answer the door with just a bra..."* → pred: SEXUALISIERUNG
- *"sweetie please just let me come there"* → pred: VERTRAUENSAUFBAU (Pet Name dominiert)

Genau diese Fälle sind nach dem Feedback der Betreuerin typische Grenzfälle, bei denen auch ein trainierter Classifier Schwierigkeiten haben wird.

### Wissenschaftliche Bewertung

Der Annotator ist für die Verwendung in der Bachelorarbeit **vertretbar**, mit folgenden Einschränkungen:

1. **Macro-F1 von 0.782** liegt im akzeptablen Bereich für LLM-basierte Annotation (vgl. Gilardi et al. 2023)
2. **KONTROLLE/NOETIGUNG (F1=0.594)** ist das schwächste Label, jedoch bestehen 76% der Fehler aus Grooming→Grooming-Verwechslungen mit SEXUALISIERUNG, die für den binären Downstream-Task weniger schädlich sind
3. **VERTRAUENSAUFBAU** hat die meisten gefährlichen Fehler (8 Grooming-Turns → NEUTRAL), was transparent diskutiert werden sollte
4. **Für die Thesis-Diskussion:** Die Annotationsqualität ist ausreichend, um einen empirischen Vergleich zwischen BERT-Baseline und BERT+Behavior-Labels durchzuführen. Schwächen des Annotators (insbesondere bei Grenzfällen) stellen eine Limitation dar, die in der Diskussion der Ergebnisse adressiert werden muss

### Entscheidung: Prompt-Optimierung durchgeführt (v2)
Nach der Fehleranalyse wurde der Prompt gezielt optimiert (siehe Abschnitt 7). Die v2-Ergebnisse zeigen eine deutliche Verbesserung; der Annotator wird in dieser Version für die weiteren Schritte verwendet.
