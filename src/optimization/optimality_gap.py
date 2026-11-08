from typing import Dict, Any, Tuple
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import BESSConfig
from src.optimization.industrial_milp import IndustrialBESSOptimizerMILP
from src.utils.industrial_profile import generate_industrial_profiles
from src.ml.price_forecaster import run_epf_experiment


def run_optimality_gap_study(
    train_end_date: str = "2024-06-30",
    test_start_date: str = "2024-07-01",
    test_end_date: str = "2024-07-31",
    contract_power_mw: float = 3.0,
    penalty_per_mw_exceeded: float = 1000.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Kluczowy eksperyment badawczy pracy magisterskiej:
    Ocena zysków i luki efektywności (Optimality Gap) dla przedsiębiorstwa przemysłowego.
    """
    print("================================================================================")
    print("  BADANIE NAUKOWE: ANALIZA OPTIMALITY GAP DLA PRZEDSIĘBIORSTWA PRZEMYSŁOWEGO")
    print("================================================================================")

    # 1. Krok ML: Wygenerowanie prognoz cen na zbiorze testowym
    df_ml_forecast, ml_metrics = run_epf_experiment(
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )

    actual_prices = df_ml_forecast["actual_price_rce_pln"]
    predicted_prices = df_ml_forecast["predicted_price_rce_pln"]
    timestamps = df_ml_forecast.index
    dt = 0.25

    # 2. Generowanie profilu przemyslowego (Taryfa B23) i PV
    df_industry = generate_industrial_profiles(
        timestamps=timestamps,
        contract_power_mw=contract_power_mw,
    )

    load = df_industry["industrial_load_mw"].to_numpy()
    pv = df_industry["pv_generation_mw"].to_numpy()
    cap_fee = df_industry["capacity_fee_pln_mwh"].to_numpy()

    # 3. Konfiguracja BESS i optymalizatora przemyslowego
    cfg = BESSConfig.from_json()
    opt = IndustrialBESSOptimizerMILP(config=cfg, dt_hours=dt)

    # --- STRATEGIA 1: BASELINE (BEZ BESS) ---
    p_grid_base = np.maximum(0.0, load - pv)
    energy_cost_base = p_grid_base * actual_prices.to_numpy() * dt
    cap_cost_base = p_grid_base * cap_fee * dt
    over_contract_base = np.maximum(0.0, p_grid_base - contract_power_mw) * penalty_per_mw_exceeded * dt
    total_cost_base = float(np.sum(energy_cost_base + cap_cost_base + over_contract_base))
    peak_grid_base = float(np.max(p_grid_base))

    # --- STRATEGIA 2: ORACLE (PERFECT FORESIGHT NA RZECZYWISTYCH CENACH) ---
    print("\n[OPTIMALITY GAP] Rozwiazywanie modelu Oracle (Perfect Foresight)...")
    res_oracle, kpis_oracle = opt.optimize(
        prices=actual_prices,
        load_mw=load,
        pv_mw=pv,
        capacity_fee_pln_mwh=cap_fee,
        contract_power_mw=contract_power_mw,
    )
    total_cost_oracle = kpis_oracle["total_procurement_cost_pln"]
    peak_grid_oracle = kpis_oracle["max_grid_peak_mw"]
    savings_oracle = total_cost_base - total_cost_oracle

    # --- STRATEGIA 3: ML-DRIVEN DISPATCH (PLANOWANIE NA PROGNOZACH LIGHTGBM) ---
    print("[OPTIMALITY GAP] Rozwiazywanie planu pracy BESS na prognozach ML...")
    res_ml_plan, _ = opt.optimize(
        prices=predicted_prices,
        load_mw=load,
        pv_mw=pv,
        capacity_fee_pln_mwh=cap_fee,
        contract_power_mw=contract_power_mw,
    )

    # Wykonanie planu BESS (p_ch, p_dis) w zakladzie i rozliczenie po CENACH RZECZYWISTYCH
    p_ch_ml = res_ml_plan["p_charging_mw"].to_numpy()
    p_dis_ml = res_ml_plan["p_discharging_mw"].to_numpy()
    p_grid_ml = np.maximum(0.0, load - pv + p_ch_ml - p_dis_ml)

    energy_cost_ml = p_grid_ml * actual_prices.to_numpy() * dt
    cap_cost_ml = p_grid_ml * cap_fee * dt
    over_contract_ml = np.maximum(0.0, p_grid_ml - contract_power_mw) * penalty_per_mw_exceeded * dt
    deg_cost_ml = cfg.cycle_cost_pln_per_mwh * p_dis_ml * dt

    total_cost_ml = float(np.sum(energy_cost_ml + cap_cost_ml + over_contract_ml + deg_cost_ml))
    peak_grid_ml = float(np.max(p_grid_ml))
    savings_ml = total_cost_base - total_cost_ml

    # --- WYZNACZENIE WSKAZNIKA OPTIMALITY GAP ---
    gap_percent = ((savings_oracle - savings_ml) / savings_oracle) * 100.0 if savings_oracle > 0 else 0.0
    efficiency_captured = (savings_ml / savings_oracle) * 100.0 if savings_oracle > 0 else 0.0

    print("\n======================= WYNIKI BADANIA EMPIRYCZNEGO =======================")
    print(f"Horyzont testowy:                        {test_start_date} do {test_end_date}")
    print(f"Liczba kwadransow (T):                   {len(timestamps)}")
    print(f"Koszt zaopatrzenia BASELINE (Bez BESS):   {total_cost_base:,.2f} PLN")
    print(f"Koszt zaopatrzenia ORACLE (Teoretyczny): {total_cost_oracle:,.2f} PLN  (Oszczednosc: {savings_oracle:,.2f} PLN)")
    print(f"Koszt zaopatrzenia ML-DRIVEN (Realny):   {total_cost_ml:,.2f} PLN  (Oszczednosc: {savings_ml:,.2f} PLN)")
    print("---------------------------------------------------------------------------")
    print(f"Wskaznik Optimality Gap:                 {gap_percent:.2f} % (Luka efektywnosci)")
    print(f"Stopien przechwycenia potencjalu (EPR):  {efficiency_captured:.2f} %")
    print(f"Szczyt poboru z sieci (Peak Shaving):    {peak_grid_base:.2f} MW -> {peak_grid_ml:.2f} MW (Limit: {contract_power_mw:.2f} MW)")
    print("===========================================================================\n")

    # Zapis tabeli LaTeX
    latex_table = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\caption{Wpływ magazynu BESS i prognoz ML na koszty zaopatrzenia zakładu przemysłowego}\n"
        "\\label{tab:optimality_gap_industrial}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "\\textbf{Strategia} & \\textbf{Koszt całk. [PLN]} & \\textbf{Oszczędność [PLN]} & \\textbf{Redukcja [\\%]} & \\textbf{Szczyt [MW]} \\\\\n"
        "\\hline\n"
        f"1. Baseline (Brak BESS)       & {total_cost_base:,.2f} & 0.00 & 0.00\\% & {peak_grid_base:.2f} \\\\\n"
        f"2. Oracle (Perfect Foresight) & {total_cost_oracle:,.2f} & {savings_oracle:,.2f} & {savings_oracle/total_cost_base*100:.2f}\\% & {peak_grid_oracle:.2f} \\\\\n"
        f"3. ML-Driven (LightGBM EPF)   & {total_cost_ml:,.2f} & {savings_ml:,.2f} & {savings_ml/total_cost_base*100:.2f}\\% & {peak_grid_ml:.2f} \\\\\n"
        "\\hline\n"
        f"\\multicolumn{{5}}{{l}}{{\\textbf{{Optimality Gap:}} {gap_percent:.2f}\\% \\quad \\textbf{{Efektywność modelu ML:}} {efficiency_captured:.2f}\\%}} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    latex_file = PROJECT_ROOT / "data" / "results" / "table_optimality_gap.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    df_results = pd.DataFrame(
        {
            "actual_price_pln": actual_prices,
            "predicted_price_pln": predicted_prices,
            "industrial_load_mw": load,
            "pv_generation_mw": pv,
            "p_grid_baseline_mw": p_grid_base,
            "p_grid_oracle_mw": res_oracle["p_grid_mw"].to_numpy(),
            "p_grid_ml_mw": p_grid_ml,
            "p_ch_ml_mw": p_ch_ml,
            "p_dis_ml_mw": p_dis_ml,
            "cost_step_baseline_pln": energy_cost_base + cap_cost_base + over_contract_base,
            "cost_step_oracle_pln": res_oracle["total_cost_pln"].to_numpy(),
            "cost_step_ml_pln": energy_cost_ml + cap_cost_ml + over_contract_ml + deg_cost_ml,
        },
        index=timestamps,
    )

    out_file = PROJECT_ROOT / "data" / "results" / "optimality_gap_study.csv"
    df_results.to_csv(out_file)
    print(f"[OPTIMALITY GAP] Zapisano pelne wyniki do: {out_file}")
    print(f"[OPTIMALITY GAP] Zapisano tabele LaTeX do: {latex_file}")

    kpis = {
        "cost_baseline_pln": total_cost_base,
        "cost_oracle_pln": total_cost_oracle,
        "cost_ml_pln": total_cost_ml,
        "savings_oracle_pln": savings_oracle,
        "savings_ml_pln": savings_ml,
        "optimality_gap_percent": gap_percent,
        "efficiency_captured_percent": efficiency_captured,
        "peak_grid_baseline_mw": peak_grid_base,
        "peak_grid_oracle_mw": peak_grid_oracle,
        "peak_grid_ml_mw": peak_grid_ml,
    }

    return df_results, kpis


if __name__ == "__main__":
    run_optimality_gap_study()
