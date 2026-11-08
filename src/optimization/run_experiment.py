from pathlib import Path
from typing import Optional
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import BESSConfig
from src.optimization.bess_milp import BESSOptimizerMILP

PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_market_prices.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def run_arbitrage_simulation(
    start_date: str = "2024-07-01",
    end_date: str = "2024-07-31",
    output_filename: str = "simulation_results.csv",
    config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Uruchamia pelna symulacje optymalizacyjna na rzeczywistych danych PSE.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = BESSConfig.from_json(config_path)

    print(f"[SIMULATION] Wczytuje dane rynkowe z: {PROCESSED_FILE}")
    df = pd.read_csv(PROCESSED_FILE, index_col=0, parse_dates=True)

    # Filtrowanie wybranego okna czasowego (w czasie lokalnym lub UTC)
    # Wybieramy zakres dat
    df_filtered = df.loc[start_date:end_date]
    if df_filtered.empty:
        raise ValueError(f"Brak danych dla przedzialu {start_date} do {end_date}")

    T = len(df_filtered)
    print(f"[SIMULATION] Rozpatrywany horyzont: {start_date} do {end_date} (kwadransow: {T}, dni: {T/96:.1f})")

    optimizer = BESSOptimizerMILP(config=config, dt_hours=0.25)
    print("[SIMULATION] Uruchamiam solver MILP (HiGHS)...")
    res_df, kpis = optimizer.optimize(df_filtered["rce_pln"])

    # Dolaczenie informacji lokalnych
    if "dtime_local" in df_filtered.columns:
        res_df["dtime_local"] = df_filtered["dtime_local"]

    output_path = RESULTS_DIR / output_filename
    res_df.to_csv(output_path, index=True)

    print("\n================ PODSUMOWANIE EKONOMICZNE (KPI) ================")
    print(f"Status rozwiazania:           {kpis['status']}")
    print(f"Laczny zysk netto:            {kpis['total_profit_pln']:,.2f} PLN")
    print(f"Sredni zysk dzienny:          {kpis['total_profit_pln'] / (T/96):,.2f} PLN/dzien")
    print(f"Calkowita energia naladowana: {kpis['total_energy_charged_mwh']:.2f} MWh")
    print(f"Calkowita energia oddana:     {kpis['total_energy_discharged_mwh']:.2f} MWh")
    print(f"Straty na sprawnosci (RT):    {kpis['efficiency_losses_mwh']:.2f} MWh")
    print(f"Liczba pelnych cykli EFC:     {kpis['equivalent_full_cycles']:.2f}")
    print(f"Srednia cena ladowania:       {kpis['avg_charging_price_pln']:.2f} PLN/MWh")
    print(f"Srednia cena rozladowania:    {kpis['avg_discharging_price_pln']:.2f} PLN/MWh")
    print(f"Sredni spread arbitrazowy:    {kpis['avg_discharging_price_pln'] - kpis['avg_charging_price_pln']:.2f} PLN/MWh")
    print("=================================================================\n")
    print(f"[SIMULATION] Wyniki zapisano do: {output_path}")

    return res_df


if __name__ == "__main__":
    # Testowy przebieg na 1 miesiac (lipiec 2024 - czeste ceny ujemne w poludnie!)
    run_arbitrage_simulation(start_date="2024-07-01", end_date="2024-07-31")
