import sys
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import BESSConfig
from src.optimization.bess_milp import BESSOptimizerMILP

st.set_page_config(
    page_title="BESS Arbitrage Optimization | SGH Visual Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_market_prices.csv"


@st.cache_data(show_spinner=False)
def load_market_data():
    if not PROCESSED_FILE.exists():
        st.error(f"Nie znaleziono pliku {PROCESSED_FILE}. Uruchom najpierw preprocess_data.py!")
        st.stop()
    df = pd.read_csv(PROCESSED_FILE, index_col=0, parse_dates=True)
    return df


def main():
    st.title("⚡ Optymalizacja Portfela Zakupowego Energii (BESS & MILP)")
    st.caption("Praca Magisterska SGH | Promotor: dr Mariusz Kozakiewicz | Visual Analytics & Numerical Optimization")

    # Wczytanie danych
    with st.spinner("Wczytywanie znormalizowanych szeregow czasowych PSE..."):
        df_all = load_market_data()

    min_ts = df_all.index.min().to_pydatetime()
    max_ts = df_all.index.max().to_pydatetime()

    # --- SIDEBAR: PARAMETRY I KONTROLA ---
    st.sidebar.header("⚙️ Parametry BESS")
    nominal_cap = st.sidebar.number_input("Pojemnosc nominalna [MWh]", min_value=0.5, max_value=100.0, value=4.0, step=0.5)
    max_power = st.sidebar.number_input("Maksymalna moc [MW]", min_value=0.25, max_value=50.0, value=2.0, step=0.25)
    rte = st.sidebar.slider("Sprawnosc cyklu (Round-Trip) [%]", min_value=60, max_value=100, value=90, step=1) / 100.0
    cycle_cost = st.sidebar.number_input("Koszt degradacji [PLN/MWh]", min_value=0.0, max_value=200.0, value=0.0, step=5.0)

    with st.sidebar.expander("Zaawansowane limity SoC", expanded=False):
        min_soc = st.slider("Min SoC [%]", min_value=0, max_value=40, value=10, step=5) / 100.0
        max_soc = st.slider("Max SoC [%]", min_value=60, max_value=100, value=90, step=5) / 100.0
        init_soc = st.slider("Poczatkowy SoC [%]", min_value=int(min_soc * 100), max_value=int(max_soc * 100), value=50, step=5) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.header("📅 Okno Czasowe Symulacji")
    default_start = datetime.date(2024, 7, 1)
    default_end = datetime.date(2024, 7, 7)
    
    date_range = st.sidebar.date_input(
        "Zakres dat",
        value=(default_start, default_end),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

    tz_option = st.sidebar.radio("Strefa czasowa wykresow", ["Czas lokalny (Europe/Warsaw)", "Czas uniwersalny (UTC)"])

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Wybierz poprawny zakres dat (poczatek i koniec) w menu bocznym.")
        st.stop()

    start_date, end_date = date_range

    # Filtrowanie danych
    mask = (df_all.index.date >= start_date) & (df_all.index.date <= end_date)
    df_window = df_all.loc[mask]

    if df_window.empty:
        st.warning("Brak danych w wybranym przedziale czasowym.")
        st.stop()

    st.sidebar.write(f"Wybrano **{len(df_window)}** kwadransow ({len(df_window)/96:.1f} dni).")

    # Inicjalizacja konfiguracji
    config = BESSConfig(
        nominal_capacity_mwh=nominal_cap,
        max_charging_power_mw=max_power,
        max_discharging_power_mw=max_power,
        round_trip_efficiency_ratio=rte,
        initial_soc_ratio=init_soc,
        min_soc_ratio=min_soc,
        max_soc_ratio=max_soc,
        cycle_cost_pln_per_mwh=cycle_cost,
    )

    # --- URUCHOMIENIE OPTYMALIZACJI ---
    optimizer = BESSOptimizerMILP(config=config, dt_hours=0.25)
    
    with st.spinner("Rozwiazywanie problemu MILP za pomoca SciPy HiGHS..."):
        res_df, kpis = optimizer.optimize(df_window["rce_pln"])

    # Przygotowanie osi czasu do prezentacji
    if tz_option.startswith("Czas lokalny"):
        time_index = df_window.index.tz_convert("Europe/Warsaw")
        x_axis = [t.strftime("%Y-%m-%d %H:%M") for t in time_index]
        x_title = "Czas lokalny (Europe/Warsaw)"
    else:
        x_axis = df_window.index
        x_title = "Czas UTC"

    res_df["display_time"] = x_axis

    # --- KARTY WSKAZNIKOW (KPIs) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Laczny zysk netto", f"{kpis['total_profit_pln']:,.2f} PLN")
    c2.metric("Sredni zysk dzienny", f"{kpis['total_profit_pln'] / max(1, len(df_window)/96):,.2f} PLN/dzien")
    spread = kpis['avg_discharging_price_pln'] - kpis['avg_charging_price_pln']
    c3.metric("Przechwycony spread", f"{spread:,.2f} PLN/MWh")
    c4.metric("Pelne cykle (EFC)", f"{kpis['equivalent_full_cycles']:.2f}")
    c5.metric("Straty RT", f"{kpis['efficiency_losses_mwh']:.2f} MWh")

    st.markdown("---")

    # --- WYKRES 1: DISPATCH SCHEDULE & PRICES ---
    st.subheader("1. Harmonogram pracy BESS na tle cen krańcowych RCE (PSE)")
    fig1 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cena energii RCE [PLN/MWh]", "Moc BESS: Ladowanie (+) / Rozladowanie (-) [MW]"),
        row_heights=[0.5, 0.5]
    )

    # Ceny
    fig1.add_trace(
        go.Scatter(
            x=res_df["display_time"],
            y=res_df["price_rce_pln"],
            mode="lines",
            name="Cena RCE",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=1, col=1
    )
    # Linia zera na cenach
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Moc ladowania (dodatnia)
    fig1.add_trace(
        go.Bar(
            x=res_df["display_time"],
            y=res_df["power_charging_mw"],
            name="Ladowanie (P_ch)",
            marker_color="#2ca02c",
            opacity=0.85
        ),
        row=2, col=1
    )
    # Moc rozladowania (ujemna dla przejrzystosci visual analytics)
    fig1.add_trace(
        go.Bar(
            x=res_df["display_time"],
            y=-res_df["power_discharging_mw"],
            name="Rozladowanie (P_dis)",
            marker_color="#d62728",
            opacity=0.85
        ),
        row=2, col=1
    )

    fig1.update_layout(
        height=550,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig1.update_xaxes(title_text=x_title, row=2, col=1)
    fig1.update_yaxes(title_text="PLN / MWh", row=1, col=1)
    fig1.update_yaxes(title_text="Moc [MW]", row=2, col=1)
    st.plotly_chart(fig1, use_container_width=True)

    # --- WYKRES 2: STATE OF CHARGE & CUMULATIVE PROFIT ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("2. Stan naładowania magazynu (SoC)")
        fig_soc = go.Figure()
        fig_soc.add_trace(
            go.Scatter(
                x=res_df["display_time"],
                y=res_df["soc_ratio"] * 100,
                fill="tozeroy",
                mode="lines",
                name="SoC [%]",
                line=dict(color="#9467bd", width=2),
                fillcolor="rgba(148, 103, 189, 0.2)"
            )
        )
        # Limity SoC
        fig_soc.add_hline(y=min_soc * 100, line_dash="dot", line_color="red", annotation_text=f"Min SoC ({min_soc*100:.0f}%)")
        fig_soc.add_hline(y=max_soc * 100, line_dash="dot", line_color="orange", annotation_text=f"Max SoC ({max_soc*100:.0f}%)")
        fig_soc.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis=dict(title="SoC [%]", range=[0, 105]),
            xaxis=dict(title=x_title)
        )
        st.plotly_chart(fig_soc, use_container_width=True)

    with col_right:
        st.subheader("3. Skumulowany zysk z arbitrażu (Cash Flow)")
        fig_profit = go.Figure()
        fig_profit.add_trace(
            go.Scatter(
                x=res_df["display_time"],
                y=res_df["cumulative_profit_pln"],
                mode="lines",
                name="Zysk skumulowany [PLN]",
                line=dict(color="#2ca02c", width=2.5)
            )
        )
        fig_profit.update_layout(
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
            yaxis=dict(title="PLN"),
            xaxis=dict(title=x_title)
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # --- TABELA I EKSPORT DANYCH ---
    with st.expander("🔍 Podgląd szczegółowych danych kwadransowych i eksport CSV", expanded=False):
        st.dataframe(res_df, use_container_width=True)
        csv_data = res_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Pobierz wyniki jako CSV",
            data=csv_data,
            file_name=f"bess_dispatch_{start_date}_{end_date}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
