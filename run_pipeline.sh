#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

source .venv/bin/activate
export PYTHONPATH=.

echo "=== [1/3] ETL: Przetwarzanie i normalizacja szeregow czasowych PSE ==="
python3 src/etl/preprocess_data.py

echo "=== [2/3] Testy jednostkowe bilansu i fizyki BESS (pytest) ==="
pytest -v

echo "=== [3/3] Badanie naukowe: Machine Learning vs Oracle (Optimality Gap) ==="
python3 src/optimization/optimality_gap.py

echo "=== Pipeline wykonany pomyslnie! Wyniki w data/results/ ==="
