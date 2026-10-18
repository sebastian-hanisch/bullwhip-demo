# Bullwhip-Effekt: Wer zahlt für die Glättung? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bullwhip-demo.streamlit.app/)**

Interaktive Fall-Demo (Bestandsmanagement, Lieferkette): In einer Kette Einzelhandel → Großhandel → Distributor → Fabrik schwankt der Kundenbedarf nur leicht, die **Bestellungen weiter oben aber immer stärker**: Jede Stufe prognostiziert aus den Bestellungen der Stufe darunter
und bestellt nach oben, und die Schwankung schaukelt sich auf – der **Bullwhip-Effekt**. Bekannt ist die Formel; die Demo fragt nach dem, was sie nicht sagt: **Was kosten Glättung, geteilte Bedarfsinformation, kürzere Wiederbeschaffungszeit und längere Prognosefenster an Kettenbestand, wenn der
Servicegrad in jeder Stufe gleich bleibt – und wer zahlt?** Live auf **einem Verlauf über 300 Perioden** (alle sieben Maßnahmen über denselben Endkundenbedarf: Bestellungen, Lager, Verstärkung je Stufe gegen die exakte Theorie, PDF der Auswertung) und **vorgerechnet** über 500 Wiederholungen je Zelle
bei gleichem Servicegrad, die die Kostenaussage tragen.

Teil des Portfolios für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. **Fall-Demo der Themenseite „Bestandsmanagement“** neben `inventory-routing-demo` (Bestand beim Kunden mit Touren). Alle Verläufe sind deterministisch (Seed), die Streuung kommt aus der Stichprobe.

## Warum dieses Problem

Im Portfolio gab es kein Lieferketten- und kein Regelungsthema; das Kernmodell (Prognose-Rückkopplung in einer Kette von Bestellstufen mit physischer Lagerführung) ist neu. Die Multi-Agenten-Linie (`contract-net-demo`, `task-swap-demo`, `dcop-demo`, `marl-demo`, `mappo-demo`) ist thematisch verwandt
(dezentrale Akteure mit lokaler Information), aber mathematisch verschieden: dort Kran-Auftragsvergabe, hier Bestandsdynamik. Nächste Nachbarn sind die Zeitreihen-Linie (`naive-forecast-demo`, `exponential-smoothing-demo`: der gleitende Durchschnitt ist dort ein Prognoseverfahren, hier die
Rückkopplungsschleife) und `inventory-routing-demo` (Schwester der Bestandsmanagement-Seite).

**Ehrlich zur Erwartbarkeit:** Der Bullwhip-Effekt ist Lehrbuchstoff (Forrester, Lee, Padmanabhan und Whang, Chen und Kollegen). Die Formel zu zeigen wäre ein Beer Game mit Formel. Der eigene Aufhänger dieser Demo ist deshalb die **Kosten- und Anreizsicht bei gleichem Servicegrad**: die naheliegende
Erwartung „Glättung erhöht die Kosten“ stimmt nur, wenn man den Servicegrad nicht festhält; bei festgehaltenem Servicegrad spart Glättung Kettenbestand, aber **die glättende Stufe zahlt selbst**, und der Nutzen fällt bei den Stufen darüber an – ein Anreizproblem, das die Verstärkungskennzahl nicht zeigt.
Chen et al. (2000) wurden nicht nachgeschlagen: die Zuordnung der Formel zur Originalarbeit ist durch Herleitung und Simulation belegt (Befund 1), das Paper selbst nicht gelesen.

## Befunde und Korrekturen gegenüber dem Plan

Der Detailplan (`plan_bullwhip`) und die Vorab-Messreihe (`messreihe_bullwhip`) sind verbindlich; beim Bau haben sich diese Punkte ergeben:

- **AP 0 (a) – Zellen-Abdeckung.** Die Regler stehen auf gemessenen Stufen (3 · 3 · 3 · 4 = 108 Kombinationen aus CV, ρ, L und p). Die Messreihe variiert je Zelle **einen** Parameter ausgehend vom Hauptszenario (CV 0,2, ρ 0, L 2, p 10); nur dieses hat alle 13 Maßnahmen, die **neun** (nicht acht, wie der Plan
  schreibt) übrigen Zellen (ρ 0,5, ρ 0,8, L 1, L 4, p 6, p 20, p 40, CV 0,1, CV 0,3) haben Basis, Glättung γ 0,5 und Info teilen. Deshalb liegen nur **10 von 108** Kombinationen genau auf einer gemessenen Zelle (30 weichen in einem Parameter ab, 44 in zweien, 24 in dreien). Die App zeigt für alle
  anderen die nächstliegende gemessene Zelle mit ausdrücklichem Hinweis (Abstand = Summe der Stufenabstände; bei Gleichstand gewinnt die Zelle, deren variierter Parameter in der Reihe L, p, ρ, CV weiter vorn steht). Das z je Stufe für die Live-Kette gehört zur **Konfiguration** (CV, ρ, effektives L und p, Regel, γ),
  nicht zur Maßnahme: „L halbiert“ bei L = 4 ist die Konfiguration L = 2 und nimmt das z des Hauptszenarios. Damit haben **46 von 1341** (Kombination × Glättung × Maßnahme)-Fällen ein genau gemessenes z; für den Rest kommt z aus der nächstliegenden gemessenen Konfiguration (mit Hinweis) oder aus der groben
  Live-Kalibrierung. Eine Zusatzmessung (zum Beispiel CV 0,3 mit p 6) wurde nicht gemacht: der Hinweis genügt.
- **AP 0 (b) – Anzeige-Seeds.** Alle Nachfrage-Seeds der Messreihe liegen bei 10000 und darüber (Kalibrierung Seed s, Auswertung s + 1, Wiederholung s + 777); die Anzeige-Seeds der Presets liegen bei 0 bis 299. `tools/tune_presets.py` durchsucht 200 bis 299 nach Verläufen, die für alle fünf Presets die qualitativen
  Tageskriterien erfüllen (`bw_stories.day_criteria`): 82 von 100 Seeds erfüllen alle fünf; gewählt wurde **Seed 202**, dessen Standardverlauf mit −15,8 % genau auf dem Messreihen-Mittel (−15,7 %) liegt.
- **AP 0 (c) – Schnellkalibrierung.** `tools/quick_check.py` (12 Seeds je Maßnahme, Hauptszenario): die Schnellkalibrierung (20 × 700, ein Durchlauf) liegt im Mittel bis 0,11 (Basis, Stufe 3 und 4: 1,03 / 1,37 / 1,77 / 1,95 gegen 1,07 / 1,33 / 1,66 / 1,84 der vollen Kalibrierung) und bei Glättung γ 0,5 bis 0,25 (Stufe 4:
  3,11 gegen 3,36) neben dem vollen z; die größte Einzel-Abweichung ist 0,31, die Streuung zwischen Seeds 0,02 bis 0,08. Der Servicegrad, den das schnelle z auf einer unabhängigen Stichprobe erreicht, liegt zwischen 94,0 und 96,0 % (Ziel 95 %), die Summe Lager gegenüber der Messreihe bei +5 % (Basis 284 gegen 270), −2 % (Glättung 224 gegen 228) und +2 %
  (Info teilen 201 gegen 197). **Grob, aber brauchbar** für eine Live-Einschätzung; die App kennzeichnet sie als „grob“ und die Messreihe bleibt die Aussage. Eine Kalibrierung dauert 2 bis 4 Sekunden (die App rechnet alle bis zu sieben Maßnahmen, mit Zeitlimit 90 s und Pause 4 s).
- **Die Meldung folgt der gewählten Maßnahme.** Der Plan nennt drei Zustände für die Glättung („Glättung senkt hier den Kettenbestand“ / „bringt hier wenig“ / „kostet hier“). Weil die Kennzahlen eine „gewählte Maßnahme“ zeigen, gibt es eine reine Anzeigewahl (keine Regler der Rechnung), und die Meldung nennt deren Namen; für die Glättung ist der Wortlaut genau der des Plans.
  Wo eine Maßnahme nur im Hauptszenario gemessen ist (nur Einzelhandel, Info teilen + Glättung, L halbiert, Fenster verdoppelt), sagt die App das.
- **Standardfehler-Bedingung im Urteil.** Der Plan nennt Schwellen (≤ −5 %, > +2 %); zusätzlich gilt (wie bei den Presets) ein Vorzeichen nur ab 2 Standardfehlern. An den gemessenen Zellen ändert das nichts (Standardfehler der Änderung 0,1 bis 0,4 Prozentpunkte).
- **Korrektur einer Zahl der Messreihe.** ERGEBNIS.md nennt für das Fenster 64 mit Niveausprüngen „+4,8 % gegen p 32“; das war aus den gerundeten Summen 176 und 168 gerechnet. Aus den ungerundeten Summen sind es **+4,6 %** (Befund 8).
- **Kurze Lieferzeit, Kriterium „Info teilen nur ≈ −10 %“** ist als Band zwischen −15 % und −5 % (und unter −2 Standardfehlern) festgelegt; gemessen −10,0 ± 0,1 %.
- **47 Checks.** Die 47 „PASS“-Zeilen der Messreihe sind 47 einzelne Tests (18 + 4 + 2 + 3 + 5 + 5 + 2 + 4 + 3 + 1); die statistischen laufen zusätzlich mit um 1000 verschobenen Seeds (Test `test_the_statistical_checks_do_not_hang_on_one_seed`), damit sie an keinem einzelnen Seed hängen.

## Modell

- **Kette.** Vier Stufen; Stufe 1 (Einzelhandel) sieht den Endkundenbedarf D<sub>t</sub> = AR(1) mit Mittel 100, Randstreuung 100 · CV und einstellbarer Autokorrelation ρ; Stufe k+1 sieht die Bestellung der Stufe k in derselben Periode; die Fabrik bestellt bei einem unbegrenzten externen Lieferanten; Lieferzeit L je Stufe (Bestellung am
  Periodenende, Ankunft zu Beginn von t + L); Backorders werden voll nachgeliefert.
- **Politik je Stufe.** Order-up-to S = m · μ̂ + z · σ̂ · √m mit Mittel und Standardabweichung der letzten p Werte der Prognose-Eingangsreihe (einschließlich des aktuellen), Schutzintervall m = L; Bestellung q = max(0; S − Bestandsposition), **nie negativ** (die Abschneidung bei 0 ist ein realer Dämpfer, den die lineare Theorie nicht kennt).
- **Maßnahmen** (immer alle sieben gerechnet): Basis; **alle glätten** (nur der Anteil γ der Bestandslücke wird korrigiert, q = max(0; μ̂ + γ · (S − IP − μ̂))); **nur Einzelhandel glättet**; **Info teilen** (alle Stufen prognostizieren aus dem Endkundenbedarf); **Info teilen + Glättung**; **L halbiert**; **Fenster verdoppelt**. „L halbiert“ bei L = 1 und
  „Fenster verdoppelt“ bei p = 40 entfallen (ohne Wirkung, mit Hinweis ausgeblendet).
- **Kosten und Servicegrad.** Bestand h = 1 und Rückstand 9 je Einheit und Periode; Servicegrad β = 1 − nicht sofort gelieferter Bedarf / neuer Bedarf der Stufe (Fill Rate).
- **Gleicher Servicegrad.** z je Stufe per Bisektion so gewählt, dass β = 95 % (oberste Stufe zuerst, zwei Durchläufe; Kalibrierung auf 200 Wiederholungen × 1200 Perioden), **ausgewertet auf einer unabhängigen Stichprobe** (500 × 1500; dort liegt β je Stufe zwischen 94,8 und 95,3 %); daneben derselbe Lauf bei **festem z = 1,645**.
- **Verstärkung** = Var(Bestellung)/Var(Bedarf) je Stufe nach 100 Perioden Einschwingen; exakte lineare Theorie (Faltung der Kettenfilter) als Kontrollzeile.

Die formale Beschreibung steht im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

Kern: `bw_model.py` (Kette, physische Lagerführung, Kalibrierung) und `bw_theory.py` (exakte lineare Theorie), **mechanisch aus `bull.py` der Messreihe übernommen, Logik unverändert**. Die Bitgleichheit ist belegt: der Wiederholungslauf des ganzen Sweeps (`tools/check_full.py`) liefert alle **8274 Zahlen** (Theorie-Blöcke und 63 Kalibrierungsjobs mit z je Stufe, Verstärkung, Lager, Servicegrad, Kosten)
**bitgleich** zu `sweep_data.json` der Messreihe (md5 der Ausgabe von `tools/sweep.py` identisch: 767a84bc5737e87b733513a31cf88e80). In der CI läuft dieselbe Prüfung auf eingefrorenen Nachfragereihen (`tests/test_frozen_reference.py`).

Die App rechnet live nur **einen** Verlauf (400 Perioden, sieben Ketten zusammen etwa 0,3 s, je Einstellung zwischengespeichert) und liest sonst `data/bw_results.json` (84 KB, erzeugt von `tools/build_results.py` aus `sweep_data.json`). Die Meldung stützt sich auf die vorgerechnete Messreihe, nicht auf den einen Verlauf.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Hauptszenario (CV 0,2, ρ 0, L 2, p 10, 4 Stufen), gleicher Servicegrad β = 95 % je Stufe, Summe Lager aller vier Stufen, 500 Wiederholungen (Standardfehler der Summen 0,1 bis 0,4), Änderung gegenüber der Basis derselben Zelle. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund |
|---|---|
| 1 · Formel gegen Simulation | Stufe 1 ist **exakt** 1 + 2m/p + 2m²/p² (m = L = Schutzintervall; 20 Kombinationen, Abweichung < 1e-12); bei mehreren Stufen ist die Produktformel **nur eine Untergrenze**: Stufe 4, m = 2, p = 5: Simulation 37,9, exakt 37,7, Produktformel 20,2; das Verhältnis exakt/Produkt liegt zwischen **1,03** (m 1, p 20) und **3,43** (m 4, p 3). Die Verstärkung hängt nur von m/p ab. Mit Sicherheitsbestand-Term und Abschneidung (m 2, p 10, CV 0,2): Stufe 4 linear 6,33, mit z = 1,645 und Abschneidung 12,56, ohne Abschneidung 14,80 (7,5 % der Stufe-4-Bestellungen sind 0). |
| 2 · L und p dominieren | Summe Lager bei L = 1 / 2 / 4: **94 / 270 / 997**; bei p = 6 / 10 / 20 / 40: **692 / 270 / 148 / 116**; ein einziger Regler verschiebt den Kettenbestand um das **10,6-fache (L)** bzw. **6,0-fache (p)**. |
| 3 · Verstärkung ist ein schwacher Kostenproxy | Rangkorrelation der Stufe-4-Verstärkung mit der Summe Lager über 13 Maßnahmen **0,45**, ohne die drei, die L oder p ändern, **0,27**. Glättung an allen Stufen (γ 0,3) drückt die Verstärkung der Fabrik um 83 % (11,1 → 1,8) und spart nur **17,3 %** Kettenbestand; „nur Fabrik glättet“: Verstärkung −72 %, Kettenbestand **+1,6 %**. Autokorrelation: die Verstärkung der linearen Theorie sinkt mit ρ, der Kettenbestand steigt aber (270 / 489 / 555 für ρ 0 / 0,5 / 0,8). |
| 4 · Fester z gegen gleicher Servicegrad | Bei **festem z = 1,645** sieht Glättung billiger aus (Summe Lager 293 → 152 bei γ 0,5), verschlechtert aber den Service drastisch (β je Stufe 92,3 / 85,2 / 83,3 / 87,8 % gegen 97,6 / 95,7 / 94,0 / 93,5 % in der Basis); Kettenkosten 472 → 634 (γ 0,5) → 883 (γ 0,3). **Bei gleichem Servicegrad** ist die Kette mit Glättung billiger (−15,7 % bei γ 0,5, −17,3 % bei γ 0,3). Die Erwartung „Glättung erhöht die Kosten“ stimmt nur ohne festgehaltenen Servicegrad. |
| 5 · Wer zahlt | Die glättende Stufe braucht selbst mehr Lager: Einzelhändler **+7,1 / +16,2 / +29,0 %** bei alle glätten mit γ 0,7 / 0,5 / 0,3, **+36,5 %** bei „nur Einzelhandel glättet, γ 0,3“; der Nutzen fällt oben an: Fabrik −25,6 / −30,2 / −37,3 % bzw. −42,3 %. Nutzen und Kosten liegen in verschiedenen Unternehmen: ein Anreizproblem. Glätten nur an der Quelle schlägt Glätten überall: **−30,8 %** gegen −17,3 % (γ 0,3); jede glättende Stufe braucht ein größeres z (1,39 / 3,26 / 4,82 / 4,26 gegen 1,46 / 2,05 / 2,48 / 2,21) und zahlt diesen Preis, an vier Stellen also viermal. |
| 6 · Maßnahmen im Überblick | Änderung der Summe Lager: L halbiert **−65,1 %**, Fenster verdoppelt −45,4 %, Info teilen **−26,9 %**, Info teilen + Glättung (γ 0,5) −42,9 %, L halbiert + Info teilen −68,7 % (Zusatznutzen des Teilens bei kurzer L nur 3,6 Prozentpunkte). Wiederholung mit anderen Seeds: alle 13 Summen innerhalb −0,4 bis +0,5 %, Rangfolge identisch (Rangkorrelation 1,000). |
| 7 · Regime | **Glättung ist regimeabhängig:** ρ 0 / 0,5 / 0,8: −15,7 / −5,4 / **+6,9 %**; L 1 / 2 / 4: +0,6 / −15,7 / −4,4 %; p 6 / 10 / 20 / 40: **−47,4** / −15,7 / −0,7 / −1,8 %; CV 0,1 / 0,2 / 0,3: −4,4 / −15,7 / −21,3 %. **Info teilen hilft überall, unterschiedlich stark** (m/p): ρ: −26,9 / −35,7 / −36,0 %; L: −10,0 / −26,9 / −49,3 %; p: **−52,6** / −26,9 / −7,3 / **−1,9 %**; CV: −24,8 / −26,9 / −29,6 %. Vorbehalt: bei festem z verschlechtert das Teilen den Service oben (β 97,4 / 94,1 / 91,7 / 92,0 %): die Sicherheitsbestandsrechnung muss folgen (z an der Fabrik 2,38 statt 1,84). |
| 8 · Fenster mit Niveausprüngen | Summe Lager über p = 4 / 8 / 16 / 32 / 64: i.i.d. **1807 / 390 / 169 / 123 / 107** (monoton), mit zufälligen Niveausprüngen (±20 % im Mittel alle 50 Perioden) **1844 / 438 / 202 / 168 / 176**: Minimum bei p = 32, bei p = 64 wieder **+4,6 %**. |

## Ehrliche Grenzen

- **Lehrbuchthema.** Formel und Grundeffekt sind bekannt; neu ist die Kosten- und Anreizsicht bei gleichem Servicegrad, in diesem Modell, bei diesen Reglern. Alle Aussagen gelten „in diesem Modell, bei diesen Reglern“.
- **Stilisierte Kette:** vier Stufen mit gleichen L und p (in den Checks auch ungleiche), ein Produkt, unbegrenzter externer Lieferant, keine Kapazitätsgrenzen, keine Losgrößen, keine Informationsverzögerung zwischen den Stufen, keine Rationierung bei Engpässen. Parameter erfunden, nicht kalibriert; das Servicegrad-Ziel β = 95 % in **allen**
  Stufen ist eine Modellwahl.
- **Varianzkosten nicht bewertet:** Kapazitäts-, Umrüst- und Expresskosten der Bestellvarianz fehlen; der Nutzen der Glättung für die oberen Stufen ist deshalb eher unterschätzt. Die Kosten von L-Verkürzung, Datenaustausch und längeren Fenstern ebenfalls nicht.
- **Kleine Fenster** (p 4 bis 6) erzeugen durch die σ-Schätzung extreme Verstärkung und Bestand (p 4: Stufe 4 bis 342): ein Schätzartefakt der Regel, kein Naturgesetz; Basisfenster deshalb p = 10, das Preset „Kleines Fenster“ (p = 6) mit Hinweis.
- **Sicherheitsbestandsrechnung** aus der Ein-Perioden-Streuung desselben Fensters (mit einer Prognosefehler-Streuung ändern sich die absoluten Zahlen); **Glättung** als proportionaler Regler (andere Formen nicht getestet); Politik: gleitender Durchschnitt und Order-up-to.
- **Regime-Zellen einmal gerechnet** (nur das Hauptszenario zusätzlich mit anderen Seeds): kleine Unterschiede nicht deuten.
- **Ein Verlauf streut.** Die Live-Zahlen tragen die Überschrift „ein Verlauf“ (Summe Lager typisch ±5 % um die Messreihe, Änderungen durch Maßnahmen ±5 Prozentpunkte); die Aussage trägt die Messreihe.
- **Zellen-Abdeckung und Schnellkalibrierung** siehe AP 0 (a) und (c): 10 von 108 Reglerkombinationen sind genau gemessen, das Live-z ist sonst die nächstliegende gemessene Konfiguration oder grob (Servicegrad ±1 Punkt, Summe Lager ±5 %).

## Tests

Die Suite `python -m pytest tests/ -v` (Anzahl und Laufzeit: siehe unten) besteht aus:

- **Korrektheit des Kerns** (`test_checks.py`, `test_model_units.py`, `test_frozen_reference.py`): die 47 Checks der Messreihe in voller Größe (Simulation == exakte Theorie auf 4 Standardfehler + 1 %; Stufe 1 exakt 1 + 2m/p + 2m²/p² für 20 Kombinationen; Bestellungen == elementweise Faltung; Handinstanz p = 2, L = 1, Spitze +10: q = 115, 100, 95; konstanter Bedarf ⇒ konstante Bestellungen und β = 1; Bestandspositions-Identität und
  Warenerhaltung mit bindender Abschneidung; einstufig gegen Newsvendor-Analytik; Zweig-Tests je Maßnahme; Kalibrierung; Wiederholbarkeit), Handrechnungen mit einem Zufallsstrom-Ersatz und die **Bitgleichheit auf eingefrorenen Nachfragereihen** gegen die Ergebnisse von `bull.py` (11 Fälle, 2 Kalibrierungen).
- **Ergebnisdatei und Auswertung** (`test_results.py`, `test_claims.py`, `test_stories.py`, `test_preset_stories.py`): Klassifikation der Jobs, Zell-Zuordnung, Urteil an den Schwellen (jede Bedingung einzeln), z-Zuordnung, Nachrechnung der Theorie-Blöcke, **jede Zahl dieses READMEs**, Presets an künstlichen Werten (jedes Kriterium einzeln gekippt) und die Abnahme der echten Presets.
- **Live-Verlauf** (`test_live.py`): Lagerverlauf je Periode gegen `bw_model.physical`, Kontrollzeile gegen die exakte Theorie, z-Herkunft, Schnellkalibrierung (Zeitlimit mit Ersatzuhr, Pause, Fortschritt).
- **App** (`test_app.py`, Streamlit-AppTest): Skelett und Footer, jedes Preset, Permalink, alle Regler an Minimum und Maximum, alle drei Meldungszustände, ausgeblendete Maßnahmen, Servicegrad-Modus, Schnellkalibrierung, PDF, keine wirkungslosen Regler, Texte. Anzeige, Figuren, PDF und Werkzeuge in `test_ui_panel.py`, `test_visualization.py`, `test_pdf_export.py`, `test_tools.py`.
- **CI-Robustheit:** die CI installiert immer das neueste NumPy; Referenzwerte laufen deshalb über **eingefrorene Nachfragereihen** (`tests/data/`), statistische Tests mit Standardfehler-Toleranz und um 1000 verschobenen Seeds; keine Wall-Clock-Assertions; Live-Rechnung im AppTest auf der eingefrorenen Reihe.
- **Fehler-Einbau-Test** (`tools/mutation_check.py`, Mutantenliste in `tools/mutants.py`): 844 maschinell erzeugte Mutanten über alle Module (Vergleichsoperatoren, Plus/Minus, and/or, True/False, min/max, Zahlen +1 bzw. +10 %), jeder in einer eigenen Kopie des Projekts mit Baseline-Selbstprüfung der unveränderten Kopie (inklusive README, `data/`, `tools/`, Schutz per Umgebungsvariable `BW_MUTATION_RUN`); **794 von den Tests gefunden, 50 überlebend**. Der erste Lauf (852 Mutanten) fand 582 und ließ 270 überleben; die echten Lücken (Zahlenformate, Tabellen-Zeichenketten, Grenzfälle der Kriterien, Werkzeug-Rahmen) sind mit Tests geschlossen (`test_ui_golden.py`, `test_tools_frame.py`, `test_format.py` u. a.). Die 50 Überlebenden wurden einzeln angesehen und in `tools/mutants.py` (`EQUIVALENT_NOTES`) in fünf Gruppen begründet (gleichwertiger Kern wie die Richtung des Ringpuffers, Tie-Break-Werte der Zell-Zuordnung, Vergleiche am Nullpunkt, PDF-Seitenumbruch, Werkzeug-Rahmen); das ist eine **Einschätzung, kein Beweis** der Gleichwertigkeit.

**564 Tests, alle grün** (219 s auf dem Entwicklungsrechner, davon etwa 160 s die AppTests; ohne AppTests etwa 70 s). Volles Bau-Gate `tools/check_full.py` lokal: 47 von 47 Checks und die Wiederholung des ganzen Sweeps bitgleich (etwa 110 s), nicht in der CI.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt: Hauptansicht (live), Kernabschnitt ② (vorgerechnet), Ansichten, Expander, Footer |
| `bw_constants.py` | Regler-Stufen, Maßnahmen, Presets, Farben, feste Parameter |
| `bw_presets.py` | Regler-Spezifikation, Permalink (Begrenzen und Einrasten), Presets, Seed-Knopf |
| `bw_model.py`, `bw_theory.py` | Kette, Lagerführung, Kalibrierung und exakte lineare Theorie: mechanisch aus `bull.py` der Messreihe, Logik unverändert |
| `bw_live.py` | Live-Verlauf: alle Maßnahmen, Lagerverlauf je Periode, Kontrollzeile, z-Zuordnung, Schnellkalibrierung |
| `bw_results.py` | Laden und Auswerten von `data/bw_results.json`: Klassifikation, Zell-Zuordnung, Urteil, z je Konfiguration, Tabellen der Kernabschnitte, Abdeckung |
| `bw_stories.py` | Abnahmekriterien der Presets |
| `bw_visualization.py`, `bw_ui_panel.py`, `bw_pdf_export.py`, `bw_format.py` | Figuren, Panel und Tabellen, Auswertungs-PDF (fpdf2, griechische Buchstaben ausgeschrieben), Zahlenformate |
| `data/bw_results.json` | Aggregate der Messreihe (63 Kalibrierungsjobs mit z je Stufe, Theorie-Blöcke E1 bis E3; klein, keine Rohdaten je Wiederholung) |
| `tools/sweep.py`, `tools/report.py` | Reproduktion der Messreihe (Sweep 103 bis 107 s auf 12 Prozessen; nicht in der CI) |
| `tools/build_results.py`, `tools/freeze_reference.py`, `tools/tune_presets.py`, `tools/quick_check.py` | Ergebnisdatei bauen, Testreferenz einfrieren, Preset-Seeds suchen, Schnellkalibrierung prüfen |
| `tools/check_full.py`, `tools/mutation_check.py`, `tools/gen_mutants.py`, `tools/mutants.py` | volles Bau-Gate, Fehler-Einbau-Test mit Mutantenliste |
| `tests/` | Tests (siehe oben), `tests/data/` die eingefrorenen Nachfragereihen und Referenzwerte |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Abhängigkeiten (ungepinnt, `>=`), CI (bei Push und wöchentlich) |

## Bewusst nicht umgesetzt

Kettenlängen außer vier; ungleiche Wiederbeschaffungszeiten und Fenster je Stufe (in den Checks gerechnet, nicht in der App); andere Prognoseverfahren (exponentielle Glättung, Trend); Losgrößen und Kapazitätsgrenzen; Rationierung bei Engpässen; Informationsverzögerung zwischen den Stufen; die Kosten der Maßnahmen selbst (Datenaustausch,
Lieferzeitverkürzung); Kalibrierung an echten Lieferketten. Alles als Ausbau denkbar.

## Reproduktion der Messreihe

```bash
python tools/sweep.py                # E1 bis E3 und 63 Kalibrierungsjobs -> tools/_out/sweep_data.json (103 bis 107 s auf 12 Prozessen)
python tools/report.py               # Textbericht -> tools/_out/sweep_report.txt
python tools/build_results.py        # -> data/bw_results.json (ohne Argumente aus ../bestand-planung/messreihe_bullwhip/sweep_data.json)
python tools/check_full.py           # volles Bau-Gate: 47 Checks und Wiederholung des ganzen Sweeps gegen die Messreihe (etwa 2 Minuten)
```

`tools/_out/` steht in der `.gitignore`. Die Ausgabe von `tools/sweep.py` ist **byte-identisch** zu `sweep_data.json` der Messreihe (md5 767a84bc5737e87b733513a31cf88e80); dieselbe NumPy-Version (2.5.3) wie bei der Messreihe vorausgesetzt (mit einer anderen Version können sich die Zufallsströme ändern, die Invarianten nicht).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `python -m pytest tests/ -v`. Volles Bau-Gate: `python tools/check_full.py`. Preset-Abstimmung: `python tools/tune_presets.py [erster_Seed] [letzter_Seed] [--jobs N]`. Schnellkalibrierung prüfen: `python tools/quick_check.py`. Fehler-Einbau: `python tools/mutation_check.py [Modul] [--jobs N]`.

## Verwandte Demos mit demselben mathematischen Modell

Stand 2026-09-25. Kein Portfolio-Stück teilt das Modell (Prognose-Rückkopplung in einer mehrstufigen Bestellkette mit physischer Lagerführung); geteilt sind Nachbarschaften:

- **Zeitreihen-Linie** (`naive-forecast-demo`, `exponential-smoothing-demo`): der gleitende Durchschnitt über p Werte ist dort ein **Prognoseverfahren** („über ein paar Wochen mitteln“), hier die **Rückkopplungsschleife**, die den Bullwhip-Effekt erzeugt. Das Stück „Prognose → Bestand“ der Zeitreihen-Linie (`forecast-inventory-demo`) überschneidet sich mit dem
  Einstufen-Fall dieser Kette (Newsvendor-Analytik: einstufig mit großem Fenster stimmt die Simulation mit der Analytik überein, Test 6).
- **`inventory-routing-demo`:** Schwester auf der Bestandsmanagement-Seite (Bestand beim Kunden, Touren, Bündeln gegen reaktiv); dort die Frage „wer gehört heute auf die Tour“, hier „wer zahlt für die Glättung“.
- **Multi-Agenten-Linie** (`contract-net-demo`, `task-swap-demo`, `dcop-demo`, `marl-demo`, `mappo-demo`): thematisch verwandt (dezentrale Akteure, lokale Information; geteilte Information als Gegenstück zu „dezentral gegen zentral“), mathematisch keine Gemeinsamkeit (Kran-Auftragsvergabe).
- **`network-flow-demo`, `leercontainer-demo`:** ähnliche Kulisse (mehrstufiges Distributionsnetz, Bestandsausgleich), aber ein zentral geplanter deterministischer Fluss ohne Bestellpolitik, Prognosefehler und Dynamik über Perioden.

---

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zum Thema: [Bestandsmanagement optimieren](https://sebastianhanisch.net/bestandsmanagement-optimierung.html).
