from typing import List, Tuple
import numpy as np
import pandas as pd


def build_epf_features(
    df: pd.DataFrame,
    price_col: str = "rce_pln",
    horizon_steps: int = 96,  # 24 godziny w przod (96 x 15 min)
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Konstruuje macierz cech dla prognozowania cen energii (Electricity Price Forecasting - EPF)
    z pelnym zachowaniem rygoru braku wycieku danych (No Data Leakage).

    Wszystkie opoznienia (lagi) sa przesuniete o co najmniej `horizon_steps` (96 kwadransow),
    co odzwierciedla realny proces rynkowy planowania doby biezacej na dobe nastepna (Day-Ahead).
    """
    df_feat = df.copy()

    # Upewnienie sie co do typu indeksu czasowego
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        raise TypeError("Indeks DataFrame musi byc typu pd.DatetimeIndex!")

    # Cechy kalendarzowe i cykliczne (w strefie lokalnej Europe/Warsaw)
    local_idx = df_feat.index.tz_convert("Europe/Warsaw") if df_feat.index.tz else df_feat.index
    hour = local_idx.hour + local_idx.minute / 60.0
    dayofweek = local_idx.weekday
    month = local_idx.month

    df_feat["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df_feat["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df_feat["sin_dayofweek"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df_feat["cos_dayofweek"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df_feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    df_feat["is_weekend"] = (dayofweek >= 5).astype(int)
    # Godziny szczytowe w Polsce: dni robocze 07:00 - 22:00
    df_feat["is_peak_hours"] = ((dayofweek < 5) & (hour >= 7.0) & (hour < 22.0)).astype(int)

    # Cechy autoregresyjne (Lags) z uwzglednieniem horyzontu Day-Ahead (min. 96 krokow wstecz)
    # lag_96: ten sam kwadrans wczoraj
    # lag_192: ten sam kwadrans 2 dni temu
    # lag_672: ten sam kwadrans tydzien temu (ten sam dzien tygodnia)
    df_feat["lag_96"] = df_feat[price_col].shift(horizon_steps)
    df_feat["lag_192"] = df_feat[price_col].shift(horizon_steps + 96)
    df_feat["lag_672"] = df_feat[price_col].shift(horizon_steps + 576)

    # Cechy kroczace (Rolling stats) z historii dostepnej przed horyzontem planowania
    shifted_price = df_feat[price_col].shift(horizon_steps)
    df_feat["rolling_mean_24h"] = shifted_price.rolling(96, min_periods=96).mean()
    df_feat["rolling_std_24h"] = shifted_price.rolling(96, min_periods=96).std()
    df_feat["rolling_min_24h"] = shifted_price.rolling(96, min_periods=96).min()
    df_feat["rolling_max_24h"] = shifted_price.rolling(96, min_periods=96).max()
    df_feat["rolling_mean_7d"] = shifted_price.rolling(672, min_periods=96).mean()

    feature_cols = [
        "sin_hour", "cos_hour",
        "sin_dayofweek", "cos_dayofweek",
        "sin_month", "cos_month",
        "is_weekend", "is_peak_hours",
        "lag_96", "lag_192", "lag_672",
        "rolling_mean_24h", "rolling_std_24h",
        "rolling_min_24h", "rolling_max_24h",
        "rolling_mean_7d",
    ]

    # Usuniecie poczatkowych wierszy NaN wynikajacych z lagowania (ok. 7 dni historii)
    df_feat = df_feat.dropna(subset=feature_cols)

    return df_feat, feature_cols
