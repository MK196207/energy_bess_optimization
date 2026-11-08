from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

from src.utils.config import BESSConfig


class IndustrialBESSOptimizerMILP:
    """
    Dedykowany model MILP dla przedsiebiorstwa przemyslowego z magazynem energii BESS i fotowoltaika PV.
    Realizuje wielocelowa optymalizacje:
    1. Minimalizacja kosztu energii z sieci (zakup po cenach RCE).
    2. Peak Shaving: ochrona mocy umownej P_contract z kara za przekroczenie (soft constraint).
    3. Redukcja oplaty mocowej (unikanie poboru w godzinach szczytowych OSD 07:00-22:00).
    4. Maksymalizacja autokonsumpcji lokalnej generacji PV.
    """

    def __init__(self, config: BESSConfig, dt_hours: float = 0.25):
        self.config = config
        self.dt = dt_hours

    def optimize(
        self,
        prices: np.ndarray | pd.Series,
        load_mw: np.ndarray | pd.Series,
        pv_mw: np.ndarray | pd.Series,
        capacity_fee_pln_mwh: np.ndarray | pd.Series,
        contract_power_mw: float = 3.0,
        penalty_per_mw_exceeded: float = 2000.0,
        enforce_final_soc: bool = True,
        initial_energy_mwh: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Rozwiazuje wezlowy model bilansu energii przedsiebiorstwa przemyslowego.
        """
        if isinstance(prices, pd.Series):
            timestamps = prices.index
            prices_arr = prices.to_numpy(dtype=float)
        else:
            timestamps = np.arange(len(prices))
            prices_arr = np.asarray(prices, dtype=float)

        load_arr = np.asarray(load_mw, dtype=float)
        pv_arr = np.asarray(pv_mw, dtype=float)
        cap_fee_arr = np.asarray(capacity_fee_pln_mwh, dtype=float)

        T = len(prices_arr)
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

        # Struktura wektora zmiennych decyzyjnych x o dlugosci 7T:
        # [0 : T]       -> p_grid    (MW, laczny pobor z sieci)
        # [T : 2T]      -> p_exceed  (MW, moc ponad moc umowna P_contract)
        # [2T : 3T]     -> p_ch      (MW, ladowanie BESS)
        # [3T : 4T]     -> p_dis     (MW, rozladowanie BESS)
        # [4T : 5T]     -> e         (MWh, stan energii w BESS)
        # [5T : 6T]     -> u_ch      (binarna: 1 jesli laduje)
        # [6T : 7T]     -> u_dis     (binarna: 1 jesli rozladowuje)
        n_vars = 7 * T

        idx_p_grid = np.arange(0, T)
        idx_p_exceed = np.arange(T, 2*T)
        idx_p_ch = np.arange(2*T, 3*T)
        idx_p_dis = np.arange(3*T, 4*T)
        idx_e = np.arange(4*T, 5*T)
        idx_u_ch = np.arange(5*T, 6*T)
        idx_u_dis = np.arange(6*T, 7*T)

        # 1. Wektor kosztow c dla minimalizacji c^T x:
        # Koszt = Sum [ (prices + cap_fee) * p_grid * dt + penalty * p_exceed * dt + c_deg * p_dis * dt ]
        c_vec = np.zeros(n_vars, dtype=float)
        c_vec[idx_p_grid] = (prices_arr + cap_fee_arr) * dt
        c_vec[idx_p_exceed] = penalty_per_mw_exceeded * dt
        c_vec[idx_p_dis] = c_deg * dt

        # 2. Granice zmiennych (Bounds)
        lb = np.zeros(n_vars, dtype=float)
        ub = np.zeros(n_vars, dtype=float)

        # p_grid: [0, 20.0 MW]
        lb[idx_p_grid] = 0.0
        ub[idx_p_grid] = 20.0

        # p_exceed: [0, 20.0 MW]
        lb[idx_p_exceed] = 0.0
        ub[idx_p_exceed] = 20.0

        # p_ch: [0, p_ch_max]
        lb[idx_p_ch] = 0.0
        ub[idx_p_ch] = p_ch_max

        # p_dis: [0, p_dis_max]
        lb[idx_p_dis] = 0.0
        ub[idx_p_dis] = p_dis_max

        # e: [e_min, e_max]
        lb[idx_e] = e_min
        ub[idx_e] = e_max

        # u_ch, u_dis: [0, 1]
        lb[idx_u_ch] = 0.0
        ub[idx_u_ch] = 1.0
        lb[idx_u_dis] = 0.0
        ub[idx_u_dis] = 1.0

        bounds = Bounds(lb, ub)

        # 3. Zmienne binarne
        integrality = np.zeros(n_vars, dtype=int)
        integrality[idx_u_ch] = 1
        integrality[idx_u_dis] = 1

        # 4. Konstrukcja macierzy ograniczen:
        # (A) Bilans wezla zakladu (T rownosci): p_grid - p_ch + p_dis = load - pv
        # (B) Bilans przekroczenia mocy umownej (T nierownosci): p_grid - p_exceed <= P_contract
        # (C) Bilans energii BESS (T rownosci)
        # (D) Warunek koncowy SoC (1 nierownosc: e_{T-1} >= e_init)
        # (E) Komplementarnosc i big-M (3T nierownosci)
        n_constraints = 3 * T + (1 if enforce_final_soc else 0) + 3 * T

        A_mat = sparse.lil_matrix((n_constraints, n_vars), dtype=float)
        b_l = np.zeros(n_constraints, dtype=float)
        b_u = np.zeros(n_constraints, dtype=float)

        row = 0

        # (A) Wezlowy bilans mocy: p_grid_t - p_ch_t + p_dis_t >= load_t - pv_t
        # Umozliwia absorpcje energii ze slonca (PV) oraz bezpieczny curtailment przy pelnej baterii
        for t in range(T):
            A_mat[row, idx_p_grid[t]] = 1.0
            A_mat[row, idx_p_ch[t]] = -1.0
            A_mat[row, idx_p_dis[t]] = 1.0
            net_demand = load_arr[t] - pv_arr[t]
            b_l[row] = net_demand
            b_u[row] = np.inf
            row += 1

        # (B) Ograniczenie mocy umownej: p_grid_t - p_exceed_t <= P_contract
        for t in range(T):
            A_mat[row, idx_p_grid[t]] = 1.0
            A_mat[row, idx_p_exceed[t]] = -1.0
            b_l[row] = -np.inf
            b_u[row] = contract_power_mw
            row += 1

        # (C) Bilans stanu naladowania BESS
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

        # (D) Koncowy stan energii
        if enforce_final_soc:
            A_mat[row, idx_e[T-1]] = 1.0
            b_l[row] = e_init
            b_u[row] = np.inf
            row += 1

        # (E) Ograniczenia komplementarnosci
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

        # 5. Wywolanie solvera HiGHS
        res = milp(c=c_vec, constraints=constraints, bounds=bounds, integrality=integrality)

        if not res.success:
            raise RuntimeError(f"Industrial MILP Solver zakonczyl sie bledem: {res.status} - {res.message}")

        sol = res.x
        p_grid_opt = sol[idx_p_grid]
        p_exceed_opt = sol[idx_p_exceed]
        p_ch_opt = sol[idx_p_ch]
        p_dis_opt = sol[idx_p_dis]
        e_opt = sol[idx_e]

        # Koszty kwadransowe
        energy_cost_step = p_grid_opt * prices_arr * dt
        cap_cost_step = p_grid_opt * cap_fee_arr * dt
        penalty_cost_step = penalty_per_mw_exceeded * p_exceed_opt * dt
        deg_cost_step = c_deg * p_dis_opt * dt
        total_cost_step = energy_cost_step + cap_cost_step + penalty_cost_step + deg_cost_step

        df_res = pd.DataFrame(
            {
                "price_rce_pln": prices_arr,
                "industrial_load_mw": load_arr,
                "pv_generation_mw": pv_arr,
                "p_grid_mw": p_grid_opt,
                "p_exceed_mw": p_exceed_opt,
                "p_charging_mw": p_ch_opt,
                "p_discharging_mw": p_dis_opt,
                "energy_stored_mwh": e_opt,
                "soc_ratio": e_opt / cfg.nominal_capacity_mwh,
                "cost_energy_pln": energy_cost_step,
                "cost_capacity_fee_pln": cap_cost_step,
                "cost_penalty_pln": penalty_cost_step,
                "cost_degradation_pln": deg_cost_step,
                "total_cost_pln": total_cost_step,
                "cumulative_cost_pln": np.cumsum(total_cost_step),
            },
            index=timestamps,
        )

        kpis = {
            "status": res.message,
            "total_procurement_cost_pln": float(df_res["total_cost_pln"].sum()),
            "energy_cost_pln": float(df_res["cost_energy_pln"].sum()),
            "capacity_fee_cost_pln": float(df_res["cost_capacity_fee_pln"].sum()),
            "penalty_cost_pln": float(df_res["cost_penalty_pln"].sum()),
            "degradation_cost_pln": float(df_res["cost_degradation_pln"].sum()),
            "max_grid_peak_mw": float(df_res["p_grid_mw"].max()),
            "total_grid_energy_mwh": float((df_res["p_grid_mw"] * dt).sum()),
            "total_bess_charged_mwh": float((df_res["p_charging_mw"] * dt).sum()),
            "total_bess_discharged_mwh": float((df_res["p_discharging_mw"] * dt).sum()),
        }

        return df_res, kpis
