from typing import Tuple
import numpy as np
import pandas as pd


def generate_industrial_profiles(
    timestamps: pd.DatetimeIndex,
    contract_power_mw: float = 3.0,
    base_load_mw: float = 2.5,
    night_load_mw: float = 1.0,
    weekend_load_mw: float = 0.8,
    pv_capacity_mwp: float = 1.5,
    capacity_fee_pln_per_mwh: float = 126.70,  # Oplata mocowa w godz. 07:00-22:00 dni robocze
    seed: int = 42,
) -> pd.DataFrame:
    """
    Syntetyzuje profil zapotrzebowania przedsiebiorstwa przemyslowego (Taryfa B23)
    oraz profil generacji z zakladowej instalacji fotowoltaicznej (PV).
    """
    rng = np.random.default_rng(seed)
    local_idx = timestamps.tz_convert("Europe/Warsaw") if timestamps.tz else timestamps

    hours = local_idx.hour + local_idx.minute / 60.0
    dayofweek = local_idx.weekday
    is_weekend = dayofweek >= 5

    T = len(timestamps)
    load = np.zeros(T, dtype=float)
    pv = np.zeros(T, dtype=float)

    for i in range(T):
        h = hours[i]
        wknd = is_weekend[i]

        # 1. Profil zapotrzebowania zakladu L_t [MW]
        if wknd:
            base = weekend_load_mw + 0.1 * np.sin(2 * np.pi * h / 24.0)
        else:
            if 6.0 <= h < 14.0:
                # I zmiana produkcyjna
                base = base_load_mw + 0.5 + 0.3 * np.sin(np.pi * (h - 6.0) / 8.0)
            elif 14.0 <= h < 22.0:
                # II zmiana produkcyjna
                base = base_load_mw + 0.2 + 0.2 * np.sin(np.pi * (h - 14.0) / 8.0)
            else:
                # Nocna dolina (utrzymanie ruchu)
                base = night_load_mw + 0.1 * np.cos(np.pi * h / 6.0)

        # Szum operacyjny (± 5-8%)
        noise = rng.normal(0, 0.08)
        load[i] = max(0.5, base + noise)

        # 2. Generacja PV [MW]
        # Kat slonca w zaleznosci od pory dnia (wschod ~5:30, zachod ~20:30 w lecie)
        if 5.5 <= h <= 20.5:
            solar_curve = np.sin(np.pi * (h - 5.5) / 15.0) ** 1.5
            # Zmiennosc zachmurzenia (cloudiness factor 0.7 - 1.0)
            cloud = 0.85 + 0.15 * np.sin(i / 16.0) + rng.uniform(-0.05, 0.05)
            cloud = np.clip(cloud, 0.2, 1.0)
            pv[i] = pv_capacity_mwp * solar_curve * cloud
        else:
            pv[i] = 0.0

    # 3. Godziny obowiazywania oplaty mocowej (dni robocze 07:00 - 22:00)
    is_capacity_peak = (~is_weekend) & (hours >= 7.0) & (hours < 22.0)
    capacity_fee = np.where(is_capacity_peak, capacity_fee_pln_per_mwh, 0.0)

    # 4. Obciazenie netto bez magazynu energii: L_net = max(0, L - PV)
    net_load_without_bess = load - pv

    df_profile = pd.DataFrame(
        {
            "industrial_load_mw": load,
            "pv_generation_mw": pv,
            "net_load_no_bess_mw": net_load_without_bess,
            "is_capacity_peak": is_capacity_peak.astype(int),
            "capacity_fee_pln_mwh": capacity_fee,
            "contract_power_mw": contract_power_mw,
        },
        index=timestamps,
    )

    return df_profile
