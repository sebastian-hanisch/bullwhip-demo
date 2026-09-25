"""Konstanten der Bullwhip-Demo (Wer zahlt für die Glättung?).

Modell und Zahlen aus messreihe_bullwhip/ (siehe bestand-planung/plan_bullwhip.html). Einstellbar sind Nachfrageschwankung, Autokorrelation, Wiederbeschaffungszeit, Prognosefenster,
Glättung, der Servicegrad-Modus und der Seed. Alle Regler stehen bewusst auf den GEMESSENEN Stufen (die Messreihe variiert je Zelle einen Parameter ausgehend vom Hauptszenario; für
Kombinationen jenseits davon zeigt die App die nächstliegende gemessene Zelle und sagt es). Es laufen IMMER alle sieben Maßnahmen, kein Regler ist je wirkungslos."""

# --- Kette ---------------------------------------------------------------------------------------------------------------------------
N_STAGES = 4
STAGE_NAMES = ("Einzelhandel", "Großhandel", "Distributor", "Fabrik")
MU = 100.0                                   # Mittel des Endkundenbedarfs (wie in der Messreihe)
H_COST, P_COST = 1.0, 9.0                    # Bestand h = 1, Rückstand p = 9 je Einheit und Periode (wie in der Messreihe)
TARGET_SERVICE = 0.95                        # Servicegrad beta je Stufe im Modus "gleicher Servicegrad"
FIXED_Z = 1.645                              # fester Sicherheitsfaktor im Vergleichsmodus
SERVICE_WARN_GAP = 0.05                      # Warnung, wenn ein Verlauf mit z aus der nächstliegenden Konfiguration mehr als 5 Prozentpunkte vom Ziel-Servicegrad entfernt liegt
ZERO_TOL = 1e-12                             # Bestellung "= 0" (Abschneidung): Zählung über eine Rauschgrenze, plattformunabhängig
BURN = 100                                   # Einschwingperioden (nicht bewertet), wie in der Messreihe
LIVE_SHOWN = 300                             # angezeigte Perioden des Live-Verlaufs
LIVE_T = BURN + LIVE_SHOWN                   # gerechnete Perioden (400)

# --- Regler --------------------------------------------------------------------------------------------------------------------------
CV_OPTIONS, CV_DEFAULT = (0.1, 0.2, 0.3), 0.2
RHO_OPTIONS, RHO_DEFAULT = (0.0, 0.5, 0.8), 0.0
LEAD_OPTIONS, LEAD_DEFAULT = (1, 2, 4), 2
WINDOW_OPTIONS, WINDOW_DEFAULT = (6, 10, 20, 40), 10
GAMMA_OPTIONS, GAMMA_DEFAULT = (0.7, 0.5, 0.3), 0.5           # Anteil der Bestandslücke, der pro Periode korrigiert wird
ZMODE_EQUAL, ZMODE_FIXED = "kalibriert", "fest"
ZMODE_OPTIONS, ZMODE_DEFAULT = (ZMODE_EQUAL, ZMODE_FIXED), ZMODE_EQUAL
ZMODE_LABELS = {ZMODE_EQUAL: "gleicher Servicegrad (kalibriert)", ZMODE_FIXED: "festes z = 1,645"}
SEED_RANGE, SEED_DEFAULT = (0, 299), 202
CACHE_ENTRIES = 24                                            # Ergebnisse je Einstellung (st.cache_data)

# Achsen der Zell-Zuordnung (Stufen der Regler plus die Fenster, die durch "Fenster verdoppelt" entstehen)
LEAD_AXIS = LEAD_OPTIONS
WINDOW_AXIS = (6, 10, 12, 20, 40, 80)

# --- Maßnahmen (immer alle sieben gerechnet) --------------------------------------------------------------------------------------------
M_BASIS, M_SMOOTH, M_RETAIL, M_INFO, M_INFO_SMOOTH, M_LEAD, M_WINDOW = ("basis", "glaettung", "einzelhandel", "info", "info_glaettung", "lead_halved", "window_doubled")
MEASURES = (M_BASIS, M_SMOOTH, M_RETAIL, M_INFO, M_INFO_SMOOTH, M_LEAD, M_WINDOW)
MEASURE_LABELS = {M_BASIS: "Basis", M_SMOOTH: "alle glätten", M_RETAIL: "nur Einzelhandel glättet", M_INFO: "Info teilen", M_INFO_SMOOTH: "Info teilen + Glättung",
                  M_LEAD: "L halbiert", M_WINDOW: "Fenster verdoppelt"}
MEASURE_SHORT = {M_BASIS: "Basis", M_SMOOTH: "alle glätten", M_RETAIL: "Einzelhandel", M_INFO: "Info teilen", M_INFO_SMOOTH: "Info + Glättung", M_LEAD: "L halbiert", M_WINDOW: "Fenster ×2"}
SMOOTH_MEASURES = (M_SMOOTH, M_RETAIL, M_INFO_SMOOTH)         # Maßnahmen, auf die der Regler gamma wirkt
MEASURE_DEFAULT = M_SMOOTH                                    # Anzeige (reine Anzeigewahl)

# --- Meldungen und Urteile ------------------------------------------------------------------------------------------------------------
LOWER_PCT = -5.0                             # "senkt den Kettenbestand": Änderung der Summe Lager höchstens -5 % ...
RAISE_PCT = 2.0                              # "kostet": Änderung über +2 % ...
SE_FACTOR = 2.0                              # ... und beide Vorzeichen nur, wenn der Betrag der Änderung mehr als 2 Standardfehler beträgt
STATE_LOWER, STATE_LITTLE, STATE_COSTS = "senkt", "wenig", "kostet"

# --- Schnellkalibrierung (grob) ----------------------------------------------------------------------------------------------------------
QUICK_R, QUICK_T, QUICK_PASSES, QUICK_ITERS, QUICK_ZMAX = 20, 700, 1, 9, 10.0
QUICK_TIME_LIMIT = 90.0                      # Sekunden: danach werden die restlichen Maßnahmen nicht mehr nachkalibriert
QUICK_COOLDOWN = 4.0                         # Sekunden zwischen zwei Aufrufen (Cloud-Rechner schonen)
QUICK_SEED_OFFSET = 100000                   # Kalibrierstichprobe: Seed + Offset (unabhängig vom angezeigten Verlauf)

# --- Farben ---------------------------------------------------------------------------------------------------------------------------
COLOR_BASE, COLOR_MEASURE, COLOR_DEMAND = "#9aa5b4", "#2a6fb0", "#1c2430"
COLOR_GOOD, COLOR_BAD, COLOR_NEUTRAL, COLOR_LIGHT, COLOR_WARN = "#2e7d4f", "#c0392b", "#9aa5b4", "#9fc2e6", "#c77700"
STAGE_COLORS = ("#2a6fb0", "#2e7d4f", "#c77700", "#7d5ba6")
CHART_HEIGHT = 380

# --- Presets (Plan Abschnitt 7; Abnahmekriterien in bw_stories.py) ---------------------------------------------------------------------
# Der Seed bestimmt nur den GEZEIGTEN Verlauf: alle Nachfrage-Seeds der Messreihe liegen bei 10000 und darüber (Kalibrierung Seed s, Auswertung s + 1, Wiederholung s + 777), die
# Anzeige-Seeds bei 0 bis 299. Seed 202 kommt aus tools/tune_presets.py (Suche über 200 bis 299): die Verläufe erfüllen für alle fünf Presets die qualitativen Kriterien aus
# bw_stories.day_criteria.
PRESETS = {
    "Standard": dict(cv=0.2, rho=0.0, lead=2, window=10, gamma=0.5, zmode=ZMODE_EQUAL, seed=SEED_DEFAULT),
    "Kurze Lieferzeit": dict(cv=0.2, rho=0.0, lead=1, window=10, gamma=0.5, zmode=ZMODE_EQUAL, seed=SEED_DEFAULT),
    "Träger Bedarf": dict(cv=0.2, rho=0.8, lead=2, window=10, gamma=0.5, zmode=ZMODE_EQUAL, seed=SEED_DEFAULT),
    "Kleines Fenster": dict(cv=0.2, rho=0.0, lead=2, window=6, gamma=0.5, zmode=ZMODE_EQUAL, seed=SEED_DEFAULT),
    "Großes Fenster": dict(cv=0.2, rho=0.0, lead=2, window=40, gamma=0.5, zmode=ZMODE_EQUAL, seed=SEED_DEFAULT),
}
PRESET_HELP = {
    "Standard": "Der Grundfall: Glättung drückt die Verstärkung an der Fabrik stark, spart bei gleichem Servicegrad aber nur wenig Kettenbestand; Info teilen und kürzere Wiederbeschaffungszeit sparen mehr.",
    "Kurze Lieferzeit": "Ist die Wiederbeschaffungszeit kurz (L = 1), gibt es fast nichts zu glätten.",
    "Träger Bedarf": "Bei stark autokorreliertem Bedarf (ρ = 0,8) kostet die Glättung Kettenbestand, Info teilen hilft dagegen stark.",
    "Kleines Fenster": "Mit kurzem Fenster (p = 6) rauscht die Prognose: Glättung und geteilte Information zahlen sich enorm aus (ein Schätzartefakt der Regel, kein Naturgesetz).",
    "Großes Fenster": "Mit langem Fenster (p = 40) ist fast nichts mehr zu holen; die Maßnahme „Fenster verdoppelt“ entfällt.",
}
