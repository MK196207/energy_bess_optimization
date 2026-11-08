import pytest
from src.utils.config import BESSConfig


def test_bess_config_defaults():
    cfg = BESSConfig(
        nominal_capacity_mwh=4.0,
        max_charging_power_mw=2.0,
        max_discharging_power_mw=2.0,
        round_trip_efficiency_ratio=0.9,
    )
    assert cfg.nominal_capacity_mwh == 4.0
    assert abs(cfg.eta_charging - 0.948683) < 1e-4
    assert abs(cfg.eta_discharging - 0.948683) < 1e-4
    assert cfg.min_energy_mwh == 0.4
    assert cfg.max_energy_mwh == 3.6
    assert cfg.initial_energy_mwh == 2.0


def test_bess_config_invalid_soc():
    with pytest.raises(ValueError):
        BESSConfig(
            nominal_capacity_mwh=4.0,
            max_charging_power_mw=2.0,
            max_discharging_power_mw=2.0,
            round_trip_efficiency_ratio=0.9,
            min_soc_ratio=0.8,
            max_soc_ratio=0.2,  # min > max
        )


def test_bess_config_initial_soc_out_of_bounds():
    with pytest.raises(ValueError):
        BESSConfig(
            nominal_capacity_mwh=4.0,
            max_charging_power_mw=2.0,
            max_discharging_power_mw=2.0,
            round_trip_efficiency_ratio=0.9,
            min_soc_ratio=0.2,
            max_soc_ratio=0.8,
            initial_soc_ratio=0.95,  # initial > max
        )
