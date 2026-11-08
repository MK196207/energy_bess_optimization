# Energy BESS Optimization

> **Praca Magisterska w Szkole Głównej Handlowej w Warszawie (SGH)**  
> **Temat**: *Optymalizacja portfela zakupowego energii przez przedsiębiorstwa przemysłowe z wykorzystaniem wielkoskalowych magazynów energii (BESS), Machine Learningu i Programowania Mieszano-Całkowitoliczbowego (MILP)*  
> **Promotor**: dr Mariusz Kozakiewicz  
> **Kluczowe filary**: Visual Analytics, metody numeryczne, inżynieria danych szeregów czasowych, programowanie liniowe i mieszane (MILP).

---

## 📌 Architektura Systemu i Metodyka Badawcza

Projekt realizuje zintegrowany łańcuch analityczno-decyzyjny dla przedsiębiorstwa przemysłowego zaopatrującego się w energię elektryczną na rynku hurtowym:

```
       [ API PSE: Rynek Bilansujący (15-min RCE) ]
                           │
                           ▼
     [ 1. Moduł ETL & Audyt Ciągłości Czasowej ]
         - Normalizacja osi czasu do bezwzględnego UTC
         - Weryfikacja ciągłości kwadransowej (freq = 15 min)
                           │
                           ▼
     [ 2. Predykcja Cen Energii (Machine Learning) ]
         - Model gradientowy LightGBM (Day-Ahead EPF)
         - Cechy autoregresyjne i cykliczne bez data leakage
                           │
                           ▼
     [ 3. Modelowanie Matematyczne MILP (HiGHS) ]
         - Bilans węzła zakładu przemysłowego (Taryfa B23)
         - Redukcja szczytów poboru mocy umownej (Peak Shaving)
         - Autokonsumpcja generacji PV i unikanie opłaty mocowej
         - Ograniczenia fizyczne BESS: SoC, C-rate, sprawność RT
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [ 4. Badanie Empiryczne ]     [ 5. Visual Analytics ]
      - Analiza Optimality Gap       - Interaktywny Dashboard R Shiny
      - Oracle vs ML vs Baseline     - bslib, Plotly, DT DataTables
```

---

## 📂 Struktura Repozytorium

```plaintext
energy_bess_optimization/
├── configs/
│   └── bess_config.json            # Parametry techniczno-ekonomiczne magazynu (MWh, MW, sprawność)
├── dashboard_r/                    # Moduł Visual Analytics w R Shiny (standard SGH)
│   ├── app.R                       # Główna aplikacja Shiny (bslib, plotly, DT)
│   └── modules/                    # Moduły analityczne R
├── data/
│   ├── raw/                        # Pobrane surowe dane rynkowe z API PSE
│   ├── processed/                  # Znormalizowane szeregi czasowe zindeksowane w UTC
│   └── results/                    # Wyniki optymalizacji, prognozy ML, tabele do pracy (LaTeX)
├── src/
│   ├── etl/                        # Skrypty ETL (pobieranie z API PSE, czyszczenie, audyt)
│   │   ├── fetch_market_data.py
│   │   ├── preprocess_data.py
│   │   └── load_config.py
│   ├── ml/                         # Prognozowanie cen energii (EPF)
│   │   ├── feature_engineering.py
│   │   └── price_forecaster.py
│   ├── optimization/               # Modele optymalizacyjne MILP
│   │   ├── bess_milp.py            # Model arbitrażu czystego
│   │   ├── industrial_milp.py      # Model bilansu węzłowego przedsiębiorstwa przemysłowego
│   │   ├── optimality_gap.py       # Badanie empiryczne: Oracle vs ML vs Baseline
│   │   └── run_experiment.py       # Symulacje eksperymentalne
│   └── utils/                      # Modele pomocnicze (profil B23, PV, konfiguracja BESS)
│       ├── config.py
│       └── industrial_profile.py
├── tests/                          # Testy jednostkowe bilansu energii i ograniczeń fizycznych
├── requirements.txt                # Zależności środowiska Python
├── run_pipeline.sh                 # Skrypt wykonania całego pipeline'u (ETL + Testy + Badanie)
├── run_dashboard_r.sh              # Uruchomienie dashboardu R Shiny
└── README.md                       # Dokumentacja projektu
```

---

## ⚙️ Rygor Metodologiczny i Numeryczny

### 1. Reżim Szeregów Czasowych (Timezone Handling)
* Całość obliczeń, indeksowania oraz modeli ML/MILP operuje **wyłącznie na czasie bezwzględnym UTC** (`DatetimeIndex`).
* Konwersja do czasu lokalnego (`Europe/Warsaw`) zachodzi **wyłącznie w warstwie prezentacyjnej** (Visual Analytics / wykresy).
* Naturalna obsługa przejść na czas letni/zimowy (DST) bez sztucznego duplikowania lub usuwania interwałów.

### 2. Sformułowanie Matematyczne BESS (MILP)
* **Krok czasowy**: $\Delta t = 0{,}25\,\text{h}$ (kwadrans rozliczeniowy).
* **Sprawność cyklu**: $\eta_{ch} = \eta_{dis} = \sqrt{\eta_{RT}}$.
* **Równanie stanu naładowania**:
  $$E_t = E_{t-1} + \left( P_{ch, t} \cdot \eta_{ch} - \frac{P_{dis, t}}{\eta_{dis}} \right) \Delta t$$
* **Węzeł przemysłowy**:
  $$P_{grid, t} - P_{ch, t} + P_{dis, t} \ge L_t - P_{PV, t}$$
  z karami za przekroczenie mocy umownej ($P_{grid, t} \le P_{contract} + P_{exceed, t}$) oraz redukcją składnika opłaty mocowej w szczycie.

---

## 🚀 Instrukcja Uruchomienia

### 1. Środowisko Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Uruchomienie pełnego potoku analitycznego
```bash
# Wykonanie ETL, audytu, testów jednostkowych oraz eksperymentu badawczego
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 3. Uruchomienie Dashboardu Visual Analytics w R Shiny
```bash
# Wymagane pakiety R: shiny, bslib, plotly, DT, data.table
chmod +x run_dashboard_r.sh
./run_dashboard_r.sh
# Aplikacja dostępna pod adresem: http://localhost:3838
```

---

## 🧪 Testy Jednostkowe
```bash
pytest -v
```
Testy weryfikują m.in. zachowanie bilansu energii magazynu, niezmienniki sprawnościowe, brak jednoczesnego ładowania i rozładowania oraz skuteczność obcinania szczytów mocy (Peak Shaving).
