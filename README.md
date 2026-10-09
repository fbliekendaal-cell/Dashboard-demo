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
/Users/bliekendaal/Desktop/Code/.venv/bin/streamlit run dashboard.py
```

Open vervolgens [http://localhost:8501](http://localhost:8501) in je browser.

---

## 📊 Inhoud Synthetische Data

De meegeleverde CSV-set bevat 300 synthetische deelnemers. De dataset bevat:
- **Demografie**: Leeftijd, geslacht, geanonimiseerde postcodes in Nederland met GPS-coördinaten.
- **Gezondheids- & Risicoscores**: BMI, Framingham hartrisico, DASS-stress/angst/depressie, leefstijlscores, slaapkwaliteit (PSQI).
- **Werkplek & ASR Indicatoren**: Werkvermogen Index (WAI), burn-out risico, vitaliteit, werktevredenheid, werkdruk, uitputting.
- **Longitudinale Metingen**: Scoreverloop en historiek (2019-2026).
- **Opdrachtgevers & Stores**: Gemiddelden en trends per bedrijfslocatie.
- **Modeldemo**: 20% van de deelnemers heeft bewust geen actuele totaalleefstijlscore,
  zodat het afhaakmodel beide doelklassen kan trainen. Aankopen hebben status `paid`.
- **Risicomigratie**: De leefstijlscore staat ook als longitudinale factorhistorie geregistreerd.

## Dashboardpagina's en filters

De sidebar biedt onder meer Overzicht, Kaarten, Variabelen verkenner, Dieet & BMI,
Longitudinale analyse, Per opdrachtgever, Producteffect, Vragenlijstgedrag, Kopersprofiel,
Datakwaliteit, statistiek en Machine Learning. De experimentele Test-pagina is alleen
beschikbaar voor admins.

Geslacht, opdrachtgever en periode zijn globale filters. Leeftijd, afdeling en functie zijn
beschikbaar onder Extra filters. De sidebar toont het aantal gefilterde deelnemers; bij kleine
selecties kunnen statistische uitkomsten minder betrouwbaar zijn.
De demo bevat 300 synthetische deelnemers; grafieken en modelmetrics zijn daarom bedoeld om
functionaliteit te demonstreren en zijn niet één-op-één vergelijkbaar met productie-uitkomsten.

De BMI-categorieën gebruiken vijf banden: ernstig ondergewicht, ondergewicht, normaal gewicht,
overgewicht en obesitas. De kaarten kunnen vanuit de demo-map opnieuw uit de synthetische
CSV-data worden gegenereerd:

```bash
python kaarten.py
```

De sidebar-knop **Herlaad synthetische demo-data** leegt de Streamlit-cache en leest de huidige
synthetische gegevens opnieuw. Nieuwe synthetische gegevens genereer je met
`python3 generate_synthetic_data.py` en exporteer je daarna met `python3 export_synthetic_csv.py`.

## Machine-learningmodellen

De ML-tab traint de modellen op de beschikbare gefilterde demo-data. Beschikbare onderdelen zijn
gebruikerssegmentatie, leefstijlscore, volgende leefstijlscore, afhaakgedrag, verbetering,
productaankopen, bloeddruk en hartrisico. De modellen en evaluatiemetrics worden dynamisch op de
dashboardpagina getoond.

Het bloeddrukmodel gebruikt een vaste beslisdrempel van `0.01` om de recall voor screening te
prioriteren. Daardoor kunnen ook veel deelnemers een signaal krijgen. Dit signaal is uitsluitend
een aanleiding voor nadere controle en is geen diagnose. De weergegeven alert rate is berekend op
de holdoutset van het model.

`rec_ls_lifestyle_score` wordt alleen gebruikt wanneer de samengestelde score in de data aanwezig
is. Ontbrekende totaalscores worden niet afgeleid door deelscores te middelen: die deelscores
kunnen verschillende eenheden hebben en er is geen gevalideerde gezamenlijke formule vastgelegd.
Het leefstijlmodel traint alleen met aanwezige doelwaarden binnen het bereik 0–10.

---

## ⚙️ Configuratie & Environment Variables (`.env`)

Instellingen zijn te beheren in het `.env` bestand:

```env
# Databron (standaard: CSV, geen databaseverbinding nodig)
DATA_SOURCE=csv
CSV_DATA_DIR=data

# Authenticatie in- of uitschakelen (false voor directe demo-toegang)
SMART_HEALTH_AUTH_ENABLED=false

# Optionele Looker Studio embed-link
LOOKER_STUDIO_EMBED_URL=

# Vertrouwde 2FA-apparaten (dagen)
AUTH_TRUST_DAYS=14

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
├── analyses.py                # Statistische analyses en longitudinale berekeningen
├── ML.py                      # Machine-learningmodellen
├── variabelen.py               # Variabelenregister en categorieën
├── categorieen.py              # Centrale categorie-definities
├── kleuren.py                  # Centrale kleur- en labeldefinities
├── kaarten.py                  # Genereert HTML-kaarten uit de ingestelde databron
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
