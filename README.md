# Smart Health Dashboard (Tijdelijke Demo & Synthetische Data)

Dit is een tijdelijke, veilige kopie van het Smart Health Dashboard voor demonstratie- en testdoeleinden aan eindgebruikers. 
Alle gevoelige/productiedata is vervangen door realistische **synthetische data** in losse CSV-bestanden onder `data/`.

---

## 🚀 Snel Starten

### 1. Repository Kopiëren & Naar de Map Gaan
```bash
cd /Users/bliekendaal/.gemini/antigravity/scratch/dashboard-demo
```

### 2. Synthetische Data Exporteren (Optioneel)
De CSV-dataset staat al klaar in `data/`. Mocht je de data opnieuw willen genereren of uitbreiden:
```bash
python3 export_synthetic_csv.py
```

### 3. Dashboard Lanceren
Start de Streamlit applicatie:
```bash
streamlit run dashboard.py
```
Of met de aanwezige virtuele omgeving:
```bash
/Users/bliekendaal/Desktop/code/.venv/bin/streamlit run dashboard.py
```

Open vervolgens [http://localhost:8501](http://localhost:8501) in je browser.

---

## 📊 Inhoud Synthetische Data

De gegenereerde database bevat 450+ geanonimiseerde deelnemers met:
- **Demografie**: Leeftijd, geslacht, geanonimiseerde postcodes in Nederland met GPS-coördinaten.
- **Gezondheids- & Risicoscores**: BMI, Framingham hartrisico, DASS-stress/angst/depressie, leefstijlscores, slaapkwaliteit (PSQI).
- **Werkplek & ASR Indicatoren**: Werkvermogen Index (WAI), burn-out risico, vitaliteit, werktevredenheid, werkdruk, uitputting.
- **Longitudinale Metingen**: Scoreverloop en historiek (2019-2026).
- **Opdrachtgevers & Stores**: Gemiddelden en trends per bedrijfslocatie.

---

## ⚙️ Configuratie & Environment Variables (`.env`)

Instellingen zijn te beheren in het `.env` bestand:

```env
# Databron (standaard: CSV, geen databaseverbinding nodig)
DATA_SOURCE=csv
CSV_DATA_DIR=data

# Authenticatie in- of uitschakelen (false voor directe demo-toegang)
SMART_HEALTH_AUTH_ENABLED=false

# Log niveau
LOG_LEVEL=INFO
```

---

## 📂 Projectstructuur

```
dashboard-demo/
├── dashboard.py               # Hoofd-Streamlit applicatie
├── visualisaties.py           # Plotly & Matplotlib grafieken logic
├── data_ingestion.py          # Data consolidatie & database ingesting
├── config.py                  # Centrale configuratie
├── export_synthetic_csv.py    # Export van synthetische data naar CSV
├── data/                      # CSV-tabellen voor de demo
├── .env                       # Omgevingsvariabelen
├── requirements.txt           # Python dependencies
└── README.md                  # Documentatie
```

---

## 💡 Git & Delen met Eindgebruikers

Om deze tijdelijke repository op GitHub / GitLab / Bitbucket te plaatsen:
```bash
git init
git add .
git commit -m "Initial commit: Tijdelijk dashboard met synthetische data"
git remote add origin <JOUW_REPO_URL>
git push -u origin main
```
