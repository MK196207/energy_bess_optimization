#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

source .venv/bin/activate
export PYTHONPATH=.

echo "Uruchamianie Dashboardu Visual Analytics Streamlit..."
streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
