from typing import Dict, Any, Tuple
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.feature_engineering import build_epf_features


class PriceForecasterLightGBM:
    """
    Model prognozowania cen krańcowych RCE na horyzont Day-Ahead z wykorzystaniem LightGBM.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 7):
        self.params = {
            "objective": "regression_l1",  # Minimalizacja MAE, odpornosc na szpilki cenowe
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_cols = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.feature_cols = list(X_train.columns)
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test[self.feature_cols])

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100.0)
        
        # Pearson correlation
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

        return {
            "mae_pln": mae,
            "rmse_pln": rmse,
            "wape_percent": wape,
            "correlation": corr,
        }


def run_epf_experiment(
    train_end_date: str = "2024-06-30",
    test_start_date: str = "2024-07-01",
    test_end_date: str = "2024-07-31",
    processed_data_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Uruchamia kompletny eksperyment uczenia maszynowego:
    1. Budowa macierzy cech bez data leakage.
    2. Podział na zbiór uczący (historyczny) i zbiór testowy (out-of-sample).
    3. Trening modelu LightGBM i predykcja.
    4. Ewaluacja metryk błędu prognostycznego.
    """
    path = processed_data_path or (PROJECT_ROOT / "data" / "processed" / "processed_market_prices.csv")
    print(f"[ML] Wczytuje dane z {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    print("[ML] Inzynieria cech szeregow czasowych...")
    df_feat, feature_cols = build_epf_features(df, price_col="rce_pln", horizon_steps=96)

    # Podzial danych
    train_mask = df_feat.index <= train_end_date
    test_mask = (df_feat.index >= test_start_date) & (df_feat.index <= test_end_date)

    X_train = df_feat.loc[train_mask, feature_cols]
    y_train = df_feat.loc[train_mask, "rce_pln"]

    X_test = df_feat.loc[test_mask, feature_cols]
    y_test = df_feat.loc[test_mask, "rce_pln"]

    if X_train.empty or X_test.empty:
        raise ValueError(f"Pusty zbior treningowy ({len(X_train)}) lub testowy ({len(X_test)})!")

    print(f"[ML] Zbior uczacy:  {len(X_train)} rekordow (do {train_end_date})")
    print(f"[ML] Zbior testowy: {len(X_test)} rekordow ({test_start_date} do {test_end_date})")

    forecaster = PriceForecasterLightGBM()
    print("[ML] Uczenie modelu LightGBM...")
    forecaster.train(X_train, y_train)

    print("[ML] Generowanie prognoz cen na zbiorze testowym...")
    y_pred = forecaster.predict(X_test)

    metrics = forecaster.evaluate(y_test.to_numpy(), y_pred)
    print("\n================ METRYKI JAKOSCI PROGNOZ CENOWYCH ================")
    print(f"MAE:         {metrics['mae_pln']:.2f} PLN/MWh")
    print(f"RMSE:        {metrics['rmse_pln']:.2f} PLN/MWh")
    print(f"WAPE:        {metrics['wape_percent']:.2f} %")
    print(f"Korelacja:   {metrics['correlation']:.4f}")
    print("===================================================================\n")

    res_df = pd.DataFrame(
        {
            "actual_price_rce_pln": y_test,
            "predicted_price_rce_pln": y_pred,
            "forecast_error_pln": y_pred - y_test,
        },
        index=X_test.index,
    )

    out_file = PROJECT_ROOT / "data" / "results" / "ml_price_forecast.csv"
    res_df.to_csv(out_file)
    print(f"[ML] Wyniki prognoz zapisano do: {out_file}")

    return res_df, metrics


if __name__ == "__main__":
    # Testowy przebieg ML na lipcu 2024 (przy treningu na danych czerwcowych)
    run_epf_experiment()
