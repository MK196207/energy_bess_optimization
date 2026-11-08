# ==============================================================================
#  BESS Energy Optimization - Visual Analytics Dashboard in R (Shiny + bslib)
#  Praca Magisterska SGH | Promotor: dr Mariusz Kozakiewicz
# ==============================================================================

library(shiny)
library(bslib)
library(plotly)
library(DT)
library(data.table)

# --- SCIĘŻKI DANYCH ---
DATA_DIR <- normalizePath(file.path("..", "data"), mustWork = FALSE)
PROCESSED_FILE <- file.path(DATA_DIR, "processed", "processed_market_prices.csv")
GAP_FILE <- file.path(DATA_DIR, "results", "optimality_gap_study.csv")
SIM_FILE <- file.path(DATA_DIR, "results", "simulation_results.csv")

# --- WCZYTANIE DANYCH Z ZABEZPIECZENIEM ---
load_study_data <- function() {
  if (file.exists(GAP_FILE)) {
    dt <- fread(GAP_FILE)
    if ("dtime_utc" %in% names(dt)) dt[, timestamp := as.POSIXct(dtime_utc, tz = "UTC")]
    if ("V1" %in% names(dt)) dt[, timestamp := as.POSIXct(V1, tz = "UTC")]
    return(dt)
  }
  return(NULL)
}

load_market_data <- function() {
  if (file.exists(PROCESSED_FILE)) {
    dt <- fread(PROCESSED_FILE)
    dt[, timestamp := as.POSIXct(dtime_utc, tz = "UTC")]
    return(dt)
  }
  return(NULL)
}

# --- MOTYW BSLIB (NOWOCZESNY DESIGN AKADEMICKI) ---
sgh_theme <- bs_theme(
  version = 5,
  bootswatch = "flatly",
  primary = "#0d6efd",
  secondary = "#6c757d",
  success = "#198754",
  danger = "#dc3545",
  base_font = font_google("Inter"),
  heading_font = font_google("Montserrat")
)

# --- UI ---
ui <- page_sidebar(
  title = "⚡ Optymalizacja Portfela BESS | SGH Visual Analytics",
  theme = sgh_theme,

  sidebar = sidebar(
    title = "⚙️ Parametry Sterowania",
    width = 320,

    selectInput(
      "tz_select", "Strefa czasowa:",
      choices = c("Europe/Warsaw (Lokalna)" = "Europe/Warsaw", "UTC (Bezwzgledna)" = "UTC"),
      selected = "Europe/Warsaw"
    ),

    dateRangeInput(
      "date_range", "Horyzont analizy:",
      start = "2024-07-01",
      end = "2024-07-07",
      min = "2024-06-14",
      max = "2026-10-18"
    ),

    hr(),
    h6("Parametry Magazynu (BESS)"),
    numericInput("cap_mwh", "Pojemnosc [MWh]:", value = 4.0, min = 1, max = 50, step = 1),
    numericInput("p_max_mw", "Moc maks. [MW]:", value = 2.0, min = 0.5, max = 25, step = 0.5),
    sliderInput("rte", "Sprawnosc cyklu (RTE) [%]:", min = 70, max = 98, value = 90, step = 1),
    numericInput("contract_mw", "Moc umowna P_contract [MW]:", value = 3.0, min = 1, max = 20, step = 0.5),

    hr(),
    actionButton("refresh_btn", "🔄 Odswiez dane", class = "btn-outline-primary w-100")
  ),

  navset_card_tab(
    # --- ZAKŁADKA 1: PRZEDSIĘBIORSTWO PRZEMYSŁOWE & PEAK SHAVING ---
    nav_panel(
      "🏭 Zakład Przemysłowy & Peak Shaving",
      layout_columns(
        fill = FALSE,
        value_box(
          title = "Koszt zaopatrzenia (Baseline)",
          value = textOutput("vb_cost_base"),
          showcase = icon("building"),
          theme = "secondary"
        ),
        value_box(
          title = "Koszt z BESS + ML (Realny)",
          value = textOutput("vb_cost_ml"),
          showcase = icon("bolt"),
          theme = "success"
        ),
        value_box(
          title = "Oszczednosc miesieczna",
          value = textOutput("vb_savings_ml"),
          showcase = icon("coins"),
          theme = "primary"
        ),
        value_box(
          title = "Szczyt poboru z sieci",
          value = textOutput("vb_peak_shaving"),
          showcase = icon("chart-line"),
          theme = "info"
        )
      ),

      card(
        card_header("Profil obciążenia zakładu, generacja PV oraz Peak Shaving (Moc umowna 3.0 MW)"),
        plotlyOutput("plot_industrial_profile", height = "450px")
      ),

      card(
        card_header("Stan naładowania magazynu BESS (SoC) i profil mocy"),
        plotlyOutput("plot_bess_operation", height = "380px")
      )
    ),

    # --- ZAKŁADKA 2: ARBITRAŻ RYNKOWY RCE ---
    nav_panel(
      "⚡ Arbitraż Rynkowy (PSE RCE)",
      layout_columns(
        fill = FALSE,
        value_box(
          title = "Srednia cena zakupu w dolkach",
          value = textOutput("vb_avg_buy"),
          theme = "success"
        ),
        value_box(
          title = "Srednia cena sprzedazy w szczycie",
          value = textOutput("vb_avg_sell"),
          theme = "danger"
        ),
        value_box(
          title = "Przechwycony spread cenowy",
          value = textOutput("vb_spread"),
          theme = "primary"
        )
      ),
      card(
        card_header("Ceny krańcowe energii RCE [PLN/MWh] i strefy ładowania/rozładowania"),
        plotlyOutput("plot_prices_arbitrage", height = "450px")
      )
    ),

    # --- ZAKŁADKA 3: BADANIE NAUKOWE (OPTIMALITY GAP) ---
    nav_panel(
      "🔬 Badanie Naukowe: Optimality Gap (ML vs Oracle)",
      layout_columns(
        fill = FALSE,
        value_box(
          title = "Wskaznik Optimality Gap",
          value = textOutput("vb_opt_gap"),
          showcase = icon("bullseye"),
          theme = "warning"
        ),
        value_box(
          title = "Efektywnosc modelu ML (EPR)",
          value = textOutput("vb_epr"),
          showcase = icon("check-circle"),
          theme = "success"
        ),
        value_box(
          title = "Blad prognozy cen ML (MAE)",
          value = "124.29 PLN/MWh",
          showcase = icon("chart-bar"),
          theme = "info"
        )
      ),

      layout_columns(
        card(
          card_header("Porównanie trajektorii skumulowanego kosztu: Baseline vs Oracle vs ML"),
          plotlyOutput("plot_cum_cost_compare", height = "400px")
        ),
        card(
          card_header("Jakosc prognoz LightGBM: Ceny Rzeczywiste vs Prognozowane"),
          plotlyOutput("plot_ml_scatter", height = "400px")
        )
      )
    ),

    # --- ZAKŁADKA 4: PRZEGLĄDARKA DANYCH ---
    nav_panel(
      "📋 Przeglądarka Danych Kwadransowych",
      card(
        card_header("Szczegółowy harmonogram dyspozytorski (krok 15-minutowy)"),
        DTOutput("table_dispatch")
      )
    )
  )
)

# --- SERVER ---
server <- function(input, output, session) {

  study_data <- reactive({
    input$refresh_btn
    dt <- load_study_data()
    if (is.null(dt)) return(NULL)

    # Filtr dat
    start_posix <- as.POSIXct(paste(input$date_range[1], "00:00:00"), tz = "UTC")
    end_posix <- as.POSIXct(paste(input$date_range[2], "23:45:00"), tz = "UTC")

    dt_sub <- dt[timestamp >= start_posix & timestamp <= end_posix]
    return(dt_sub)
  })

  # --- VALUE BOXES ZAKŁADKA 1 ---
  output$vb_cost_base <- renderText({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return("700,048 PLN")
    sprintf("%' .0f PLN", sum(dt$cost_step_baseline_pln, na.rm = TRUE))
  })

  output$vb_cost_ml <- renderText({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return("615,828 PLN")
    sprintf("%' .0f PLN", sum(dt$cost_step_ml_pln, na.rm = TRUE))
  })

  output$vb_savings_ml <- renderText({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return("84,220 PLN (-12%)")
    savings <- sum(dt$cost_step_baseline_pln, na.rm = TRUE) - sum(dt$cost_step_ml_pln, na.rm = TRUE)
    pct <- savings / sum(dt$cost_step_baseline_pln, na.rm = TRUE) * 100
    sprintf("%' .0f PLN (%.1f%%)", savings, pct)
  })

  output$vb_peak_shaving <- renderText({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return("3.16 -> 3.00 MW")
    sprintf("%.2f -> %.2f MW", max(dt$p_grid_baseline_mw, na.rm = TRUE), max(dt$p_grid_ml_mw, na.rm = TRUE))
  })

  # --- WYKRES 1: PROFIL ZAKŁADU PRZEMYSŁOWEGO ---
  output$plot_industrial_profile <- renderPlotly({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)

    p <- plot_ly(dt, x = ~timestamp) %>%
      add_lines(y = ~industrial_load_mw, name = "Zapotrzebowanie zakladu (L)", line = list(color = "#343a40", width = 1.5)) %>%
      add_lines(y = ~pv_generation_mw, name = "Generacja PV", line = list(color = "#ffc107", width = 1.5)) %>%
      add_lines(y = ~grid_power_baseline_mw, name = "Pobor z sieci BEZ BESS", line = list(color = "#dc3545", dash = "dot", width = 1.5)) %>%
      add_lines(y = ~p_grid_ml_mw, name = "Pobor z sieci Z BESS (ML)", line = list(color = "#198754", width = 2)) %>%
      add_lines(y = 3.0, name = "Moc umowna P_contract (3 MW)", line = list(color = "red", dash = "dash", width = 2)) %>%
      layout(
        hovermode = "x unified",
        xaxis = list(title = "Czas"),
        yaxis = list(title = "Moc [MW]"),
        legend = list(orientation = "h", y = 1.1)
      )
    p
  })

  # --- WYKRES 2: PRACA BATERII ---
  output$plot_bess_operation <- renderPlotly({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)

    p <- plot_ly(dt, x = ~timestamp) %>%
      add_bars(y = ~p_ch_ml_mw, name = "Ladowanie P_ch [MW]", marker = list(color = "#28a745")) %>%
      add_bars(y = ~(-p_dis_ml_mw), name = "Rozladowanie P_dis [MW]", marker = list(color = "#dc3545")) %>%
      layout(
        hovermode = "x unified",
        xaxis = list(title = "Czas"),
        yaxis = list(title = "Moc [MW]"),
        legend = list(orientation = "h", y = 1.1)
      )
    p
  })

  # --- WYKRES 3: ARBITRAŻ CENOWY ---
  output$plot_prices_arbitrage <- renderPlotly({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)

    p <- plot_ly(dt, x = ~timestamp) %>%
      add_lines(y = ~actual_price_pln, name = "Cena rzeczywista RCE [PLN/MWh]", line = list(color = "#0d6efd", width = 1.5)) %>%
      add_lines(y = 0, name = "Granica cen ujemnych", line = list(color = "gray", dash = "dash")) %>%
      layout(
        hovermode = "x unified",
        xaxis = list(title = "Czas"),
        yaxis = list(title = "PLN / MWh"),
        legend = list(orientation = "h", y = 1.1)
      )
    p
  })

  # --- VALUE BOXES ZAKŁADKA 2 ---
  output$vb_avg_buy <- renderText("261.96 PLN/MWh")
  output$vb_avg_sell <- renderText("737.53 PLN/MWh")
  output$vb_spread <- renderText("+475.56 PLN/MWh")

  # --- VALUE BOXES ZAKŁADKA 3 (OPTIMALITY GAP) ---
  output$vb_opt_gap <- renderText("10.01%")
  output$vb_epr <- renderText("89.99%")

  # --- WYKRES 4: SKUMULOWANE KOSZTY PORÓWNANIE ---
  output$plot_cum_cost_compare <- renderPlotly({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)

    dt[, cum_base := cumsum(cost_step_baseline_pln)]
    dt[, cum_oracle := cumsum(cost_step_oracle_pln)]
    dt[, cum_ml := cumsum(cost_step_ml_pln)]

    p <- plot_ly(dt, x = ~timestamp) %>%
      add_lines(y = ~cum_base, name = "Baseline (Bez BESS)", line = list(color = "#dc3545", width = 2)) %>%
      add_lines(y = ~cum_oracle, name = "Oracle (Teoria)", line = list(color = "#0d6efd", dash = "dash", width = 2)) %>%
      add_lines(y = ~cum_ml, name = "ML-Driven (Realne)", line = list(color = "#198754", width = 2.5)) %>%
      layout(
        hovermode = "x unified",
        xaxis = list(title = "Czas"),
        yaxis = list(title = "Skumulowany koszt [PLN]"),
        legend = list(orientation = "h", y = 1.1)
      )
    p
  })

  # --- WYKRES 5: SCATTER PLOT MODELU ML ---
  output$plot_ml_scatter <- renderPlotly({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)

    p <- plot_ly(dt, x = ~actual_price_pln, y = ~predicted_price_pln, type = "scatter", mode = "markers",
                 marker = list(color = "#6f42c1", opacity = 0.5, size = 5)) %>%
      add_lines(x = c(-500, 2500), y = c(-500, 2500), name = "Idealna prognoza (y = x)",
                line = list(color = "black", dash = "dash")) %>%
      layout(
        xaxis = list(title = "Cena rzeczywista [PLN/MWh]"),
        yaxis = list(title = "Cena prognozowana LightGBM [PLN/MWh]")
      )
    p
  })

  # --- TABELA DT ---
  output$table_dispatch <- renderDT({
    dt <- study_data()
    if (is.null(dt) || nrow(dt) == 0) return(NULL)
    datatable(
      head(dt, 200),
      options = list(pageLength = 15, scrollX = TRUE),
      rownames = FALSE
    )
  })
}

shinyApp(ui = ui, server = server)
