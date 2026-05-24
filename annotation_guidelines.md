# Annotation Guidelines: Cybergrooming Behavior Labels
**Bachelorarbeit – Mustafa Ugurluer, TU Darmstadt**
**Version 1.0 – Mai 2026**

---

## 1. Einleitung und Zweck

Dieses Dokument definiert die sieben Behavior-Labels, die zur Annotation von Chat-Nachrichten im PAN12-Datensatz verwendet werden. Die Labels bilden die theoretischen Phasen des Cybergrooming-Prozesses als diskrete, operationalisierbare Kategorien ab.

Die Definitionen basieren auf einer Synthese der folgenden Primärquellen:
- O'Connell, R. (2003). *A typology of child cybersexploitation and online grooming practices.*
- Lanning, K. V. (2010). *Child Molesters: A Behavioral Analysis for Professionals Investigating the Sexual Exploitation of Children*
- Berson, I. R. (2003). *Grooming cybervictims: The psychosocial effects of online exploitation for youth.* 
- Black, P. J., Wollis, M., Woodworth, M., & Hancock, J. T. (2015). *A linguistic analysis of grooming strategies of online child sex offenders: Implications for our understanding of predatory sexual behavior in an increasingly computer-mediated world.* Child Abuse & Neglect.
- Lorenzo-Dus, N., Izura, C., & Pérez-Tattam, R. (2016). *Understanding grooming discourse in computer-mediated environments.*
- Ringenberg, T. et al. (2021). *A scoping review of child grooming strategies: pre- and post-internet.* 
- Zambrano, P. et al. (2019). *Technical Mapping of the Grooming Anatomy Using Machine Learning Paradigms: An Information Security Approach.*

---

## 2. Grundprinzipien

**2.1 Primary Label:** Jeder Turn erhält genau ein Label. Das Label, das den dominanten kommunikativen Zweck der Nachricht beschreibt.

**2.2 Perspektive:** Es wird ausschließlich die Nachricht des potenziellen Täters annotiert, nicht die des Opfers.

**2.3 Kontextunabhängigkeit:** Die Annotation basiert auf dem Turn selbst. Kontext kann zur Klärung herangezogen werden, darf aber nicht dazu führen, dass eine inhaltlich neutrale Nachricht als Grooming annotiert wird.

**2.4 Nichtlinearität:** Die Labels repräsentieren keine streng lineare Sequenz. Laut O'Connell (2003) und Lorenzo-Dus et al. (2016) können Grooming-Phasen überlappen, wiederholt auftreten oder übersprungen werden.

---

## 3. Label-Definitionen

---

### 3.1 NEUTRAL

**Definition:**
Nachrichten ohne erkennbaren Grooming-Zweck. Alltägliche, kontextfreie Konversation, die keiner Grooming-Phase zugeordnet werden kann.

**Theoretische Grundlage:**
NEUTRAL ist kein Grooming-Stage im literarischen Sinne, aber methodisch notwendig. Der PAN12-Datensatz enthält auch Konversationen zwischen Nicht-Tätern sowie Phasen innerhalb von Täter-Gesprächen, die kein Grooming Kontext haben. Zambrano et al. (2019) bezeichnen den initialen Gesprächseinstieg als *"Greetings and casual talks"*. Allerdings ist das im Grooming-Kontext bereits instrumentalisierter Erstkontakt und daher von NEUTRAL zu unterscheiden.

**Kernindikatoren:**
- Themen ohne Grooming-Relevanz: Wetter, Schule allgemein, Hobbys ohne emotionale Bindungsabsicht, Alltag
- Keine Affektmarker, keine Komplimente, keine persönlichen Fragen zur Vulnerabilität
- Keine sexuellen Inhalte, keine Geheimhaltungshinweise, keine Treffenplanung
 
**Abgrenzung:**
- ≠ VERTRAUENSAUFBAU: Sobald Affektmarker, Kosenamen oder emotionale Bindungssprache erkennbar sind → VERTRAUENSAUFBAU
- Bewertungsregel: „Könnte diese Nachricht 1:1 in einem harmlosen Gespräch zwischen Gleichaltrigen stehen?" → Wenn Ja: NEUTRAL

**Beispiele:**
- ✓ `"have u seen that new movie, Barn Animals"` → NEUTRAL (harmlose Alltagsfrage)
- ✓ `"always gotta think 4 the future lol"` → NEUTRAL
- ✗ `"ur the only person im talking 2"` → NICHT NEUTRAL → VERTRAUENSAUFBAU (Exklusivität)

---

### 3.2 VERTRAUENSAUFBAU

**Definition:**
Nachrichten, die den Aufbau einer emotionalen Bindung zum Opfer bezwecken. Der Täter etabliert eine vertrauensvolle, exklusive oder romantisch-affektive Beziehung. Entscheidend ist laut Lorenzo-Dus et al. (2016): dieser Vertrauensaufbau ist **täuschender Natur** (*deceptive trust development*) und dient der späteren Ausnutzung.

**Theoretische Grundlage:**
Alle zentralen Modelle benennen diese Phase:
- CEOP/Lanning (2010): *"Friendship"* als erste Phase
- O'Connell (2003): *"Friendship"* + *"Relationship"* (zwei aufeinanderfolgende Phasen)
- Berson (2003): *"Gaining trust"*
- Lorenzo-Dus et al. (2016): *"Deceptive trust development"* als Substage von *Entrapment*
- Zambrano et al. (2019): *"Greetings and casual talks"* (initialer Vertrauensaufbau)

**Kernindikatoren:**
- Kosenamen und Affektmarker: *"sweetie", "baby", "honey", "cutie"*
- Liebeserklärungen, Zuneigungsbekundungen
- Ausdruck von Vermissen oder Sorge um das Opfer
- Komplimente über Aussehen oder Persönlichkeit
- Betonung der Exklusivität der Beziehung (*"you're the only one"*)
- Versprechen von Fürsorge, Schutz oder Unterstützung

**Abgrenzung:**
- ≠ NEUTRAL: Jede Nachricht mit Kosenamen, Komplimenten oder emotionaler Bindungssprache → VERTRAUENSAUFBAU, auch wenn der Inhalt sonst harmlos erscheint
- ≠ KONTROLLE/NÖTIGUNG: Wenn die Zuneigung als Druckmittel eingesetzt wird (*"if you loved me you would..."*) → KONTROLLE/NÖTIGUNG; wenn Zuneigung aufgebaut wird ohne Druckkomponente → VERTRAUENSAUFBAU

**Beispiele:**
- ✓ `"i saw your pic and i just gotta say you look beautiful"` → VERTRAUENSAUFBAU
- ✓ `"i am real happy I met you too sweetheart"` → VERTRAUENSAUFBAU
- ✗ `"please dont be mad or upset i love you so much"` → Grenzfall: Liebeserklärung + Druck → primär VERTRAUENSAUFBAU wenn kein Zwang erkennbar; KONTROLLE wenn Schuldgefühl erzeugt wird

---

### 3.3 INFORMATIONSGEWINNUNG

**Definition:**
Nachrichten, die gezielt persönliche Informationen über das Opfer erfragen, um dessen Vulnerabilität, Erreichbarkeit und Isolierbarkeit einzuschätzen. Umfasst Fragen zu Aufenthaltsort, Familie, Tagesablauf, sozialer Einbettung sowie zur sexuellen Erfahrung im Sinne der Risikoabschätzung.

**Theoretische Grundlage:**
- CEOP/Lanning (2010): *"Information gathering"* als eigenständige Phase
- O'Connell (2003): *"Risk assessment"* und *"Exclusivity"* – der Täter prüft, ob das Opfer isolierbar und sexuell zugänglich ist
- Berson (2003): *"Discussing private topics"*, *"Risk assessment"*, *"Exclusivity"* – drei aufeinanderfolgende Substages
- Zambrano et al. (2019): *"Private info collection"* als zweite Phase

**Kernindikatoren:**
- Fragen nach Wohnort, Adresse, Umgebung
- Fragen nach Aufenthaltsort/Abwesenheit der Eltern oder Bezugspersonen
- Fragen nach Tagesablauf, Schulzeiten, Heimkommen
- Fragen nach Geschwistern, Familie, Wohnsituation
- Fragen zur sozialen Einbettung (Nachbarn, Freunde)
- Fragen zur sexuellen Erfahrung/Geschichte **im Kontext der Vulnerabilitätsabschätzung**

**Abgrenzungsregel INFORMATIONSGEWINNUNG vs. SEXUALISIERUNG:**
> *Wenn eine Frage mit sexuellem Inhalt dem Ziel der Vulnerabilitätsabschätzung dient (z.B. „Hat das Opfer schon Erfahrungen? Ist es isoliert genug?") → INFORMATIONSGEWINNUNG.*
> *Wenn eine Nachricht sexuellen Inhalt einführt, beschreibt oder normalisiert ohne primären Informationszweck → SEXUALISIERUNG.*

**Beispiele:**
- ✓ `"what time does your dad come home tomorrow?"` → INFORMATIONSGEWINNUNG (Erreichbarkeit)
- ✓ `"do you know what ur address is"` → INFORMATIONSGEWINNUNG (Lokalisierung)
- ✓ `"u ever do stuff with an older guy b4?"` → INFORMATIONSGEWINNUNG (sexuelle Vorerfahrung als Risk Assessment nach O'Connell)
- ✗ `"take your lil pants down"` → NICHT INFORMATIONSGEWINNUNG → SEXUALISIERUNG (kein Informationszweck)

---

### 3.4 GEHEIMHALTUNG/ISOLATION

**Definition:**
Nachrichten, die darauf abzielen, die Beziehung oder spezifische Handlungen geheim zu halten und/oder das Opfer von seiner sozialen Umgebung (Eltern, Freunde, Autoritätspersonen) zu isolieren. Ziel ist die Verhinderung von Entdeckung und Einmischung.

**Theoretische Grundlage:**
- CEOP/Lanning (2010): *"Isolation"* als explizite Phase
- Ringenberg et al. (2021): *"Isolation"* und *"Secrecy"* als getrennte, aber verwandte Phasen
- Lorenzo-Dus et al. (2016): *"Isolation"* als Substage von *Entrapment*
- Lanning (2010): *"Concealment"* als finale Sicherungsphase

**Kernindikatoren:**
- Aufforderung, Chats/Nachrichten zu löschen
- Fragen, ob jemand die Interaktion beobachten könnte
- Pläne, sich nicht entdecken zu lassen
- Aufforderung, Lügen gegenüber Eltern/Freunden zu erzählen
- Betonung der Illegalität oder des gesellschaftlich inakzeptablen Charakters der Beziehung als Geheimhaltungsmotiv
- Codewörter oder geheime Verabredungen

**Abgrenzungsregel GEHEIMHALTUNG vs. KONTROLLE/NÖTIGUNG:**
> *Wenn der Täter auf Entdeckungsvermeidung und Geheimhaltung fokussiert (primäres Ziel: nicht erwischt werden) → GEHEIMHALTUNG.*
> *Wenn der Täter das Opfer unter Druck setzt oder manipuliert, um ein bestimmtes Verhalten zu erzwingen (primäres Ziel: Kontrolle über das Opfer) → KONTROLLE/NÖTIGUNG.*

**Beispiele:**
- ✓ `"hun do you delet your talks so she can't find them?"` → GEHEIMHALTUNG
- ✓ `"yep...don't tell nobody okay..."` → GEHEIMHALTUNG
- ✓ `"we wont get caught make sure u delete this stuff"` → GEHEIMHALTUNG
- ✗ `"you won't tell anyone I am having sex with you, right baby?"` → Grenzfall: Geheimhaltungsaufforderung mit implizitem Druck; primär GEHEIMHALTUNG wenn keine explizite Drohung; KONTROLLE wenn Drohung erkennbar

---

### 3.5 SEXUALISIERUNG

**Definition:**
Nachrichten, die sexuellen Inhalt einführen, normalisieren, beschreiben oder explizit fordern. Umfasst alle Formen sexueller Kommunikation: explizite Beschreibungen, Anforderung sexueller Handlungen oder Bilder, sexuelle Bewertung des Opfers.

**Theoretische Grundlage:**
Dieses Label ist in allen zentralen Modellen vertreten:
- O'Connell (2003): *"Sexual stages"* als zentrale Phase
- Berson (2003): *"Sexual stages"*
- CEOP/Lanning (2010): *"Desensitization"* (schrittweise Normalisierung) + *"Abuse/Exploitation"*
- Ringenberg et al. (2021): *"Gradual sexualization"* – betont die **Gradualität** der Einführung sexueller Inhalte
- Zambrano et al. (2019): *"Sexual conversation or media exchange"*

**Kernindikatoren:**
- Explizite sexuelle Sprache oder Beschreibungen
- Anforderung oder Ankündigung sexueller Handlungen
- Anforderung oder Versprechen sexueller Bilder/Videos
- Sexuelle Bewertung von Körperteilen des Opfers
- Beschreibung eigener sexueller Erregung
- Sexuelle Rollenspiele oder Fantasien

**Abgrenzung:**
- ≠ INFORMATIONSGEWINNUNG: Sexuelle Fragen ohne Informationszweck → SEXUALISIERUNG (siehe Regel unter 3.3)
- ≠ KONTROLLE/NÖTIGUNG: Sexuelle Forderungen mit Druckkomponente → primär SEXUALISIERUNG solange sexueller Inhalt dominiert; KONTROLLE wenn der Zwang dominiert und sexueller Inhalt Mittel zum Zweck ist

**Beispiele:**
- ✓ `"i want a naked picture of you"` → SEXUALISIERUNG
- ✓ `"would u want me to put it in u?"` → SEXUALISIERUNG
- ✓ `"so do you know what ur sexiest panties are"` → SEXUALISIERUNG
- ✗ `"did you enjoy your first time?"` → Grenzfall: Bezieht sich auf Vergangenheit, wirkt wie Risk Assessment → nach Entscheidungsregel eher INFORMATIONSGEWINNUNG

---

### 3.6 KONTROLLE/NÖTIGUNG

**Definition:**
Nachrichten, die psychologischen Druck, emotionale Manipulation, Drohungen oder Zwang einsetzen, um das Verhalten des Opfers zu steuern. Ziel ist Compliance des Opfers, nicht primär Geheimhaltung oder Sexualisierung.

**Theoretische Grundlage:**
Dieses Label ist in der Literatur weniger konsistent als eigenständige Phase vertreten als die übrigen Labels. Kontroll- und Nötigungselemente werden in vielen Modellen in andere Phasen integriert (besonders die Sexualisierungsphase). Folgende Modelle benennen es jedoch explizit:
- Ringenberg et al. (2021): *"Coercion"* als eigenständige Phase im Pre-/Post-Internet-Modell
- Lorenzo-Dus et al. (2016): *"Compliance testing"* als Substage von *Entrapment* – der Täter testet, wie weit das Opfer bereit ist zu folgen
- CEOP/Lanning (2010): *"Manipulation"* als Phase

Wenn Nachrichten gleichzeitig Merkmale von KONTROLLE/NÖTIGUNG und einem anderen Label aufweisen, gilt: **Wenn Zwang oder Manipulation dominiert → KONTROLLE/NÖTIGUNG.**

**Kernindikatoren:**
- Emotionale Erpressung: Schuldgefühle erzeugen, Liebesentzug androhen
- Explizite Drohungen
- Passive Aggression als Druckmittel (*"I guess you don't want to talk anymore"*)
- Direkte Befehle mit Erwartung von Gehorsam (*"wear a skirt for me"*)
- Infantilisierung oder Degradierung des Opfers als Kontrollmittel

**Abgrenzungsregel KONTROLLE vs. GEHEIMHALTUNG:**
→ Siehe 3.4

**Abgrenzungsregel KONTROLLE vs. VERTRAUENSAUFBAU:**
> *Wenn Zuneigung oder Fürsorge gezeigt wird ohne Druckkomponente → VERTRAUENSAUFBAU.*
> *Wenn Zuneigung als Hebel eingesetzt wird, um das Opfer zu einer Handlung zu zwingen → KONTROLLE/NÖTIGUNG.*

**Beispiele:**
- ✓ `"will u do it the next time lil slut?"` → KONTROLLE/NÖTIGUNG (Befehl + Degradierung)
- ✓ `"your making me feel like shit now im just gonna go"` → KONTROLLE/NÖTIGUNG (emotionale Erpressung)
- ✓ `"I guess you don't want to talk to me any more"` → KONTROLLE/NÖTIGUNG (passive Aggression)
- ✗ `"you need to promise i wont get in trouble"` → GEHEIMHALTUNG (primär: Schutz vor Entdeckung, kein Verhaltenszwang gegenüber dem Opfer)

---

### 3.7 OFFLINE-ESKALATION

**Definition:**
Nachrichten, die einen physischen Kontakt anbahnen oder vorbereiten: Treffen arrangieren, Telefonnummern austauschen, Anfahrtswege besprechen, zeitliche Koordination für ein reales Treffen.

**Theoretische Grundlage:**
Konsistent über alle relevanten Modelle vertreten:
- O'Connell (2003): *"Conclusion"* – die Sequenz endet mit dem physischen Treffen
- Lorenzo-Dus et al. (2016): *"Approach (offline meeting)"* als eigenständige Phase
- Zambrano et al. (2019): *"Attempts for in-person contact"* als vierte und letzte Phase
- Pre-/Post-Internet (Ringenberg et al. 2021): Offline-Treffen als Eskalationspunkt

**Kernindikatoren:**
- Planung eines physischen Treffens (Ort, Zeit, Anreise)
- Austausch von Telefonnummern
- Aufforderung zum Anruf
- Fahrtplanung oder Ortsangaben im Kontext eines Treffens
- Bestätigung oder Änderung bereits geplanter Treffen

**Abgrenzung:**
- ≠ INFORMATIONSGEWINNUNG: Fragen nach dem Wohnort zum Zweck der Vulnerabilitätsabschätzung → INFORMATIONSGEWINNUNG. Wenn der Wohnort im Kontext der Anreise/Treffenplanung erfragt wird → OFFLINE-ESKALATION
- Entscheidungsregel: „Dient die Frage der Planung eines realen Treffens?" → Ja: OFFLINE-ESKALATION

**Beispiele:**
- ✓ `"remember u still gotta give me the address so i can get there"` → OFFLINE-ESKALATION
- ✓ `"gi'me your number I'll dail u rite now"` → OFFLINE-ESKALATION
- ✓ `"just lookin at directions to ur place"` → OFFLINE-ESKALATION
- ✗ `"do u know what ur address is of the place u live at?"` → Grenzfall: Wenn Treffenkontext fehlt → INFORMATIONSGEWINNUNG

---

## 4. Entscheidungsbaum für Grenzfälle

```
Ist die Nachricht ohne Grooming-Kontext denkbar?
├── Ja → NEUTRAL
└── Nein → Welcher Zweck dominiert?
    ├── Emotionale Bindung aufbauen → VERTRAUENSAUFBAU
    ├── Persönliche Info abfragen (Ort, Familie, Tagesablauf) → INFORMATIONSGEWINNUNG
    │   └── Sexuelle Info? → Ist es Risk Assessment? → Ja: INFORMATIONSGEWINNUNG / Nein: SEXUALISIERUNG
    ├── Entdeckung vermeiden / Geheimnis wahren → GEHEIMHALTUNG/ISOLATION
    ├── Sexuellen Inhalt einführen/beschreiben → SEXUALISIERUNG
    ├── Druck, Zwang, Manipulation ausüben → KONTROLLE/NÖTIGUNG
    └── Physisches Treffen vorbereiten → OFFLINE-ESKALATION
```

---

## 5. Bekannte Ambiguitäten und Umgang

| Grenzfall | Betroffene Labels | Entscheidungsregel |
|---|---|---|
| Sexuelle Fragen zur Erfahrungsabschätzung | INFO ↔ SEX | Primärzweck: Risikoabschätzung → INFO; Einführung sexueller Inhalte → SEX |
| Geheimhaltungsaufforderung mit Drohung | GEH ↔ KON | Drohung explizit erkennbar → KON; reine Schweigeanweisung → GEH |
| Zuneigung als Druckmittel | VER ↔ KON | Zwangskomponente erkennbar → KON; reine Zuneigung → VER |
| Adressfrage ohne Meeting-Kontext | INFO ↔ OFF | Kein Treffenkontext → INFO; Anreiseplanung erkennbar → OFF |
| Kosenamen in neutralem Kontext | NEU ↔ VER | Affektmarker vorhanden → VER, unabhängig vom sonstigen Inhalt |

---

## 6. Inter-Annotator Agreement

Vor der finalen Annotation sind die von Mustafa Ugurluer erstellten Annotationen durch eine zweite Person zu verifizieren (Mareike Bassenge).