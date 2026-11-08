from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "market_prices.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_market_prices.csv"


def preprocess_market_prices(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Standaryzuje szereg czasowy cen RCE:
    1. Oparcie indeksu wylacznie na osi bezwzglednej UTC ('dtime_utc').
    2. Usuniecie ewentualnych duplikatow.
    3. Reindeksacja do scislego kroku 15-minutowego (freq='15min').
    4. Interpolacja brakujacych punktow cenowych.
    5. Dodanie metadanych czasowych dla strefy Europe/Warsaw (do wizualizacji).
    """
    input_path = input_path or RAW_FILE
    output_path = output_path or PROCESSED_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PREPROCESS] Wczytuje dane surowe z: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Walidacja kolumn
    if "dtime_utc" not in df.columns:
        raise KeyError(f"Brak wymaganej kolumny 'dtime_utc' w pliku {input_path}")
    if "rce_pln" not in df.columns:
        raise KeyError(f"Brak wymaganej kolumny 'rce_pln' w pliku {input_path}")

    # 2. Parsowanie UTC i sortowanie
    df["dtime_utc"] = pd.to_datetime(df["dtime_utc"], utc=True)
    df["rce_pln"] = pd.to_numeric(df["rce_pln"], errors="coerce")

    # Usuniecie duplikatow w UTC
    init_len = len(df)
    df = df.drop_duplicates(subset=["dtime_utc"]).sort_values("dtime_utc")
    dups_removed = init_len - len(df)
    if dups_removed > 0:
        print(f"[PREPROCESS] Usunieto {dups_removed} zdublowanych rekordow.")

    # Ustawienie indeksu czasowego
    df = df.set_index("dtime_utc")

    # 3. Audyt spojnosci czasowej (krok 15 min)
    start_dt = df.index.min()
    end_dt = df.index.max()
    complete_index = pd.date_range(start=start_dt, end=end_dt, freq="15min", tz="UTC")

    missing_intervals = complete_index.difference(df.index)
    num_missing = len(missing_intervals)
    print(f"[AUDYT CZASOWY] Zakres: {start_dt} do {end_dt}")
    print(f"[AUDYT CZASOWY] Oczekiwana liczba interwalow 15-min: {len(complete_index)}")
    print(f"[AUDYT CZASOWY] Brakujace interwaly w surowych danych: {num_missing}")

    # Reindeksacja do idealnej siatki 15-minutowej
    df_clean = df.reindex(complete_index)

    # Interpolacja cen dla brakujacych przedzialow (zabezpieczenie ciaglosci solvera)
    if df_clean["rce_pln"].isna().any():
        nan_count = df_clean["rce_pln"].isna().sum()
        print(f"[PREPROCESS] Interpoluje {nan_count} wartosci NaN w cenach rce_pln...")
        df_clean["rce_pln"] = df_clean["rce_pln"].interpolate(method="time").bfill().ffill()

    # 4. Generowanie metadanych dla Visual Analytics (Europe/Warsaw)
    local_index = df_clean.index.tz_convert("Europe/Warsaw")
    df_clean["dtime_local"] = local_index.strftime("%Y-%m-%d %H:%M:%S")
    df_clean["hour_local"] = local_index.hour
    df_clean["day_of_week"] = local_index.day_name()
    df_clean["is_weekend"] = local_index.weekday >= 5

    # Statystyki biznesowe
    negative_prices = (df_clean["rce_pln"] < 0).sum()
    print("--------------------------------------------------")
    print(f"Statystyka cenowa (PLN/MWh):")
    print(f" - Min:    {df_clean['rce_pln'].min():.2f}")
    print(f" - Srednia:{df_clean['rce_pln'].mean():.2f}")
    print(f" - Max:    {df_clean['rce_pln'].max():.2f}")
    print(f" - Liczba interwalow z ujemna cena (okazje arbitrazowe): {negative_prices}")
    print("--------------------------------------------------")

    # Zapis
    df_clean.to_csv(output_path, index=True, index_label="dtime_utc")
    print(f"[PREPROCESS] Gotowe! Zapisano {len(df_clean)} znormalizowanych rekordow do: {output_path}")

    return df_clean


if __name__ == "__main__":
    preprocess_market_prices()
