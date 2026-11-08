#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Uruchamianie Dashboardu Visual Analytics w R Shiny (Port 3838) ==="
Rscript -e "shiny::runApp('dashboard_r', port = 3838, host = '0.0.0.0')"
