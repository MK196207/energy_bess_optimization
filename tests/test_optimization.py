import numpy as np
import pytest
from src.utils.config import BESSConfig
from src.optimization.bess_milp import BESSOptimizerMILP
from src.optimization.industrial_milp import IndustrialBESSOptimizerMILP


@pytest.fixture
def bess_config():
    return BESSConfig(
        nominal_capacity_mwh=4.0,
        max_charging_power_mw=2.0,
        max_discharging_power_mw=2.0,
        round_trip_efficiency_ratio=0.9,
        initial_soc_ratio=0.5,
        min_soc_ratio=0.1,
        max_soc_ratio=0.9,
        cycle_cost_pln_per_mwh=0.0,
    )


def test_milp_arbitrage_behavior(bess_config):
    optimizer = BESSOptimizerMILP(bess_config, dt_hours=0.25)
    
    low_prices = [100.0] * 12
    high_prices = [800.0] * 12
    prices = np.array(low_prices + high_prices)

    df_res, kpis = optimizer.optimize(prices)

    assert kpis["total_profit_pln"] > 0
    assert df_res["energy_charged_mwh"].iloc[:12].sum() > 0
    assert df_res["energy_discharged_mwh"].iloc[12:].sum() > 0

    simultaneous_active = (df_res["is_charging"] == 1) & (df_res["is_discharging"] == 1)
    assert not simultaneous_active.any()

    assert (df_res["energy_stored_mwh"] >= bess_config.min_energy_mwh - 1e-5).all()
    assert (df_res["energy_stored_mwh"] <= bess_config.max_energy_mwh + 1e-5).all()
    assert kpis["total_energy_discharged_mwh"] <= kpis["total_energy_charged_mwh"] + 1e-5


def test_industrial_milp_peak_shaving(bess_config):
    opt = IndustrialBESSOptimizerMILP(bess_config, dt_hours=0.25)
    T = 96
    prices = np.full(T, 400.0)
    # Obciazenie szczytowe 3.2 MW przekraczajace moc umowna 3.0 MW
    load = np.full(T, 2.0)
    load[30:50] = 3.2  # Szczyt poboru
    pv = np.zeros(T)
    cap_fee = np.zeros(T)

    df_res, kpis = opt.optimize(
        prices=prices,
        load_mw=load,
        pv_mw=pv,
        capacity_fee_pln_mwh=cap_fee,
        contract_power_mw=3.0,
    )

    # 1. Sprawdzenie statusu optymalizacji
    assert "Optimal" in kpis["status"]

    # 2. Bateria powinna rozladowac sie w szczycie, scinajac pobor ponizej 3.0 MW
    assert kpis["max_grid_peak_mw"] <= 3.0 + 1e-5
    assert df_res.loc[df_res.index[30:50], "p_discharging_mw"].sum() > 0
