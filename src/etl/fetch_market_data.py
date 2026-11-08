from pathlib import Path
from typing import Optional
import requests
import pandas as pd
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_FILE = RAW_DATA_DIR / "market_prices.csv"


def fetch_rce_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[Path] = None,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pobiera dane rynkowe RCE (Rynek Bilansujacy, rozliczenia 15-min) z API PSE.
    
    :param start_date: Data poczatkowa w formacie 'YYYY-MM-DD' (opcjonalna)
    :param end_date: Data koncowa w formacie 'YYYY-MM-DD' (opcjonalna)
    :param output_path: Sciezka zapisu pliku CSV (domyslnie data/raw/market_prices.csv)
    :param max_records: Opcjonalny limit rekordow (do testow)
    :return: pd.DataFrame z surowymi danymi
    """
    output_path = output_path or DEFAULT_OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://api.raporty.pse.pl/api/rce-pln"
    params = {}
    
    filters = []
    if start_date:
        filters.append(f"business_date ge '{start_date}'")
    if end_date:
        filters.append(f"business_date le '{end_date}'")
    
    if filters:
        params["$filter"] = " and ".join(filters)

    records = []
    current_url = base_url
    page = 1

    print(f"[ETL] Rozpoczynam pobieranie danych z API PSE: {base_url}")
    if params:
        print(f"[ETL] Zastosowany filtr OData: {params.get('$filter')}")

    while current_url:
        try:
            req_params = params if page == 1 else None
            response = requests.get(current_url, params=req_params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"[ETL ERROR] Blad pobierania strony {page}: {e}")
            time.sleep(2)
            break

        batch = data.get("value", [])
        if not batch:
            break

        records.extend(batch)
        print(f"[ETL] Strona {page}: pobrano {len(batch)} rekordow (lacznie w buforze: {len(records)})")

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break

        current_url = data.get("nextLink")
        page += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"[ETL] Sukces! Zapisano {len(df)} rekordow do: {output_path}")
    else:
        print("[ETL OSTRZEZENIE] Nie pobrano zadnych rekordow.")

    return df


if __name__ == "__main__":
    # Test szybkiego pobrania ostatnich 200 rekordow
    fetch_rce_data(max_records=200)
