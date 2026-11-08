from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

from src.utils.config import BESSConfig


class BESSOptimizerMILP:
    """
    Wysokowydajny optymalizator portfela BESS bazujacy na MILP (HiGHS solver).
    W pelni macierzowa konstrukcja ograniczen (SciPy Sparse CSC Matrix) bez petli Pythona w solverze.
    """

    def __init__(self, config: BESSConfig, dt_hours: float = 0.25):
        self.config = config
        self.dt = dt_hours  # Krok 15 min = 0.25 h

    def optimize(
        self,
        prices: np.ndarray | pd.Series,
        enforce_binary_exclusion: bool = True,
        enforce_final_soc: bool = True,
        initial_energy_mwh: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Rozwiazuje zadanie optymalizacji arbitrazu cenowego.

        :param prices: Wektor cen rynkowych (PLN/MWh) o dlugosci T
        :param enforce_binary_exclusion: Czy wymuszac zmienne binarne wykluczajace jednoczesne ladowanie/rozladowanie
        :param enforce_final_soc: Czy wymuszac koncowy stan energii >= stan poczatkowy
        :param initial_energy_mwh: Opcjonalne nadpisanie poczatkowej energii (domyslnie z configu)
        :return: (DataFrame z wynikami co krok t, slownik z kluczowymi KPI)
        """
        if isinstance(prices, pd.Series):
            timestamps = prices.index
            prices_arr = prices.to_numpy(dtype=float)
        else:
            timestamps = np.arange(len(prices))
            prices_arr = np.asarray(prices, dtype=float)

        T = len(prices_arr)
        if T == 0:
            raise ValueError("Wektor cen nie moze byc pusty!")

        cfg = self.config
        dt = self.dt
        eta_ch = cfg.eta_charging
        eta_dis = cfg.eta_discharging
        p_ch_max = cfg.max_charging_power_mw
        p_dis_max = cfg.max_discharging_power_mw
        e_min = cfg.min_energy_mwh
        e_max = cfg.max_energy_mwh
        e_init = initial_energy_mwh if initial_energy_mwh is not None else cfg.initial_energy_mwh
        c_deg = cfg.cycle_cost_pln_per_mwh

        # Struktura wektora zmiennych decyzyjnych x:
        # [0 : T]       -> p_ch   (MW, ciagle)
        # [T : 2T]      -> p_dis  (MW, ciagle)
        # [2T : 3T]     -> e      (MWh, stan energii, ciagle)
        # [3T : 4T]     -> u_ch   (binarne ladowanie)
        # [4T : 5T]     -> u_dis  (binarne rozladowanie)
        n_vars = 5 * T

        # 1. Wektor kosztow c dla minimalizacji c^T x:
        # Zysk = Sum [ (prices - c_deg) * p_dis * dt - prices * p_ch * dt ]
        # Minimalizujemy -Zysk
        c_vec = np.zeros(n_vars, dtype=float)
        c_vec[0:T] = prices_arr * dt                     # Koszt ladowania (+ w minimalizacji)
        c_vec[T:2*T] = -(prices_arr - c_deg) * dt        # Przychod z rozladowania (- w minimalizacji)

        # 2. Granice zmiennych (Bounds)
        lb = np.zeros(n_vars, dtype=float)
        ub = np.zeros(n_vars, dtype=float)

        # p_ch
        lb[0:T] = 0.0
        ub[0:T] = p_ch_max

        # p_dis
        lb[T:2*T] = 0.0
        ub[T:2*T] = p_dis_max

        # e (SoC)
        lb[2*T:3*T] = e_min
        ub[2*T:3*T] = e_max

        # u_ch, u_dis
        lb[3*T:5*T] = 0.0
        ub[3*T:5*T] = 1.0

        bounds = Bounds(lb, ub)

        # 3. Typy zmiennych (integrality: 0=ciagla, 1=calkowitoliczbowa)
        integrality = np.zeros(n_vars, dtype=int)
        if enforce_binary_exclusion:
            integrality[3*T:5*T] = 1

        # 4. Konstrukcja macierzy ograniczen w formacie rzadkim (Sparse LIL/CSR)
        # Ograniczenia:
        # (A) Bilans energii: T rownan rownosciowych
        # (B) Koncowy stan energii: 1 nierownosc (e_{T-1} >= e_init)
        # (C) p_ch <= p_ch_max * u_ch  (T nierownosci)
        # (D) p_dis <= p_dis_max * u_dis (T nierownosci)
        # (E) u_ch + u_dis <= 1 (T nierownosci)

        n_constraints = T + (1 if enforce_final_soc else 0)
        if enforce_binary_exclusion:
            n_constraints += 3 * T

        A_mat = sparse.lil_matrix((n_constraints, n_vars), dtype=float)
        b_l = np.zeros(n_constraints, dtype=float)
        b_u = np.zeros(n_constraints, dtype=float)

        row = 0

        # (A) Bilans energii
        # e_0 - eta_ch * dt * p_ch_0 + (dt / eta_dis) * p_dis_0 = e_init
        # Dla t >= 1: e_t - e_{t-1} - eta_ch * dt * p_ch_t + (dt / eta_dis) * p_dis_t = 0
        idx_p_ch = np.arange(T)
        idx_p_dis = np.arange(T, 2*T)
        idx_e = np.arange(2*T, 3*T)

        for t in range(T):
            A_mat[row, idx_e[t]] = 1.0
            if t > 0:
                A_mat[row, idx_e[t-1]] = -1.0
            A_mat[row, idx_p_ch[t]] = -eta_ch * dt
            A_mat[row, idx_p_dis[t]] = dt / eta_dis

            if t == 0:
                b_l[row] = e_init
                b_u[row] = e_init
            else:
                b_l[row] = 0.0
                b_u[row] = 0.0
            row += 1

        # (B) Koncowy stan energii e_{T-1} >= e_init
        if enforce_final_soc:
            A_mat[row, idx_e[T-1]] = 1.0
            b_l[row] = e_init
            b_u[row] = np.inf
            row += 1

        # (C, D, E) Ograniczenia komplementarnosci
        if enforce_binary_exclusion:
            idx_u_ch = np.arange(3*T, 4*T)
            idx_u_dis = np.arange(4*T, 5*T)

            for t in range(T):
                # p_ch_t - p_ch_max * u_ch_t <= 0
                A_mat[row, idx_p_ch[t]] = 1.0
                A_mat[row, idx_u_ch[t]] = -p_ch_max
                b_l[row] = -np.inf
                b_u[row] = 0.0
                row += 1

                # p_dis_t - p_dis_max * u_dis_t <= 0
                A_mat[row, idx_p_dis[t]] = 1.0
                A_mat[row, idx_u_dis[t]] = -p_dis_max
                b_l[row] = -np.inf
                b_u[row] = 0.0
                row += 1

                # u_ch_t + u_dis_t <= 1
                A_mat[row, idx_u_ch[t]] = 1.0
                A_mat[row, idx_u_dis[t]] = 1.0
                b_l[row] = -np.inf
                b_u[row] = 1.0
                row += 1

        constraints = LinearConstraint(A_mat.tocsc(), b_l, b_u)

        # 5. Wywolanie solvera SciPy HiGHS
        res = milp(c=c_vec, constraints=constraints, bounds=bounds, integrality=integrality)

        if not res.success:
            raise RuntimeError(f"Solver zakonczyl sie bledem: {res.status} - {res.message}")

        # 6. Ekstrakcja wektorow rozwiazania
        sol = res.x
        p_ch_opt = sol[0:T]
        p_dis_opt = sol[T:2*T]
        e_opt = sol[2*T:3*T]
        u_ch_opt = np.round(sol[3*T:4*T]).astype(int)
        u_dis_opt = np.round(sol[4*T:5*T]).astype(int)

        # Obliczenia metryk i profilu zysku
        energy_charged_mwh = p_ch_opt * dt
        energy_discharged_mwh = p_dis_opt * dt
        net_power_mw = p_dis_opt - p_ch_opt
        cash_flow_pln = (p_dis_opt * prices_arr - p_ch_opt * prices_arr - c_deg * p_dis_opt) * dt
        soc_ratio = e_opt / cfg.nominal_capacity_mwh

        df_res = pd.DataFrame(
            {
                "price_rce_pln": prices_arr,
                "power_charging_mw": p_ch_opt,
                "power_discharging_mw": p_dis_opt,
                "net_power_mw": net_power_mw,
                "energy_charged_mwh": energy_charged_mwh,
                "energy_discharged_mwh": energy_discharged_mwh,
                "energy_stored_mwh": e_opt,
                "soc_ratio": soc_ratio,
                "is_charging": u_ch_opt,
                "is_discharging": u_dis_opt,
                "step_profit_pln": cash_flow_pln,
                "cumulative_profit_pln": np.cumsum(cash_flow_pln),
            },
            index=timestamps,
        )

        total_profit = float(df_res["step_profit_pln"].sum())
        total_charged = float(df_res["energy_charged_mwh"].sum())
        total_discharged = float(df_res["energy_discharged_mwh"].sum())
        loss_mwh = total_charged - total_discharged + (e_init - e_opt[-1])
        equivalent_cycles = total_discharged / cfg.nominal_capacity_mwh

        kpis = {
            "status": res.message,
            "total_profit_pln": total_profit,
            "total_energy_charged_mwh": total_charged,
            "total_energy_discharged_mwh": total_discharged,
            "efficiency_losses_mwh": loss_mwh,
            "equivalent_full_cycles": equivalent_cycles,
            "avg_charging_price_pln": (
                (df_res["price_rce_pln"] * df_res["energy_charged_mwh"]).sum() / total_charged
                if total_charged > 0 else 0.0
            ),
            "avg_discharging_price_pln": (
                (df_res["price_rce_pln"] * df_res["energy_discharged_mwh"]).sum() / total_discharged
                if total_discharged > 0 else 0.0
            ),
        }

        return df_res, kpis


if __name__ == "__main__":
    from src.utils.config import BESSConfig

    cfg = BESSConfig.from_json()
    optimizer = BESSOptimizerMILP(cfg)

    # Przykladowy test na 1 dobie (96 kwadransow)
    synthetic_prices = 400 + 300 * np.sin(np.linspace(0, 2 * np.pi, 96))
    df_out, kpis = optimizer.optimize(synthetic_prices)

    print("\n[MILP Test] Zakonczono rozwiazywanie modelu:")
    for k, v in kpis.items():
        if isinstance(v, float):
            print(f" - {k}: {v:.2f}")
        else:
            print(f" - {k}: {v}")
