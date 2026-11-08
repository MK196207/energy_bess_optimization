from pathlib import Path
from typing import Optional
import json
import math
from pydantic import BaseModel, Field, model_validator


class BESSConfig(BaseModel):
    """
    Parametry techniczno-ekonomiczne wielkoskalowego magazynu energii (BESS).
    """
    nominal_capacity_mwh: float = Field(..., gt=0, description="Pojemnosc nominalna w MWh")
    max_charging_power_mw: float = Field(..., gt=0, description="Maksymalna moc ladowania w MW")
    max_discharging_power_mw: float = Field(..., gt=0, description="Maksymalna moc rozladowania w MW")
    round_trip_efficiency_ratio: float = Field(..., gt=0, le=1.0, description="Sprawnosc cyklu AC-AC (eta_RT)")
    initial_soc_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Poczatkowy stan naladowania (0-1)")
    min_soc_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Minimalny dopuszczalny SoC (ochrona baterii)")
    max_soc_ratio: float = Field(0.9, ge=0.0, le=1.0, description="Maksymalny dopuszczalny SoC")
    cycle_cost_pln_per_mwh: float = Field(0.0, ge=0.0, description="Koszt degradacji / cyklu w PLN/MWh")

    @model_validator(mode="after")
    def validate_soc_bounds(self) -> "BESSConfig":
        if self.min_soc_ratio >= self.max_soc_ratio:
            raise ValueError(f"min_soc_ratio ({self.min_soc_ratio}) musi byc mniejszy niz max_soc_ratio ({self.max_soc_ratio})")
        if not (self.min_soc_ratio <= self.initial_soc_ratio <= self.max_soc_ratio):
            raise ValueError(
                f"initial_soc_ratio ({self.initial_soc_ratio}) musi zawierac sie w przedziale "
                f"[{self.min_soc_ratio}, {self.max_soc_ratio}]"
            )
        return self

    @property
    def eta_charging(self) -> float:
        """Sprawnosc ladowania: sqrt(eta_RT)"""
        return math.sqrt(self.round_trip_efficiency_ratio)

    @property
    def eta_discharging(self) -> float:
        """Sprawnosc rozladowania: sqrt(eta_RT)"""
        return math.sqrt(self.round_trip_efficiency_ratio)

    @property
    def min_energy_mwh(self) -> float:
        return self.min_soc_ratio * self.nominal_capacity_mwh

    @property
    def max_energy_mwh(self) -> float:
        return self.max_soc_ratio * self.nominal_capacity_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc_ratio * self.nominal_capacity_mwh

    @classmethod
    def from_json(cls, config_path: Optional[Path] = None) -> "BESSConfig":
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "bess_config.json"
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


if __name__ == "__main__":
    cfg = BESSConfig.from_json()
    print("Poprawnie zaladowano konfiguracje BESS:")
    print(cfg.model_dump_json(indent=2))
    print(f"eta_ch = {cfg.eta_charging:.4f}, eta_dis = {cfg.eta_discharging:.4f}")
    print(f"E_min = {cfg.min_energy_mwh:.2f} MWh, E_max = {cfg.max_energy_mwh:.2f} MWh")
