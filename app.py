import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime as dt

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

# =======================
# ⚙️ Page Config (must be first Streamlit call)
# =======================
st.set_page_config(layout="wide", page_title="Stock & News Analysis Dashboard", page_icon="📊")

# =======================
# 📂 Load Data
# =======================
@st.cache_data
def load_data():
    # Use relative paths (works both locally & on GitHub/Streamlit Cloud)
    stock_df = pd.read_csv("./data/merged_output_17_08_2025.csv", parse_dates=["Date"])
    news_df = pd.read_csv("./data/combined_output.csv", parse_dates=["date"])

    # Replace NaN with 0 / defaults
    stock_df = stock_df.fillna(0)
    news_df['stock_name'] = news_df.get('stock_name', pd.Series(dtype="object")).fillna('Nan')
    news_df = news_df.fillna(0)
    return stock_df, news_df

stock_df, news_df = load_data()

# =======================
# 📊 Aggregation + Plot Helper
# =======================
def analyze_and_plot(df, group_col, title, rename_col=None):
    avg_impact_score = []
    avg_model_confidence_score = []
    avg_senti_score = []
    group_values = []
    growth_percent = []

    # guard: if group_col missing, return empty fig
    if group_col not in df.columns:
        empty_df = pd.DataFrame(columns=[rename_col or group_col, "avg_impact_score",
                                         "avg_model_confidence_score", "avg_senti_score", "growth_percent"])
        fig = px.bar(empty_df, y=rename_col or group_col, x=["avg_impact_score",
                    "avg_senti_score", "growth_percent", "avg_model_confidence_score"],
                    orientation='h', barmode="group", title=title, height=500)
        return empty_df, fig

    for i in df[group_col].dropna().unique():
        subset = df[df[group_col] == i]

        # handle missing columns safely
        impact = subset.get('impact_score', pd.Series([0]*len(subset))).astype(float)
        conf = subset.get('confidence_score', pd.Series([0]*len(subset))).astype(float)
        senti = subset.get('sentiment_score', pd.Series([0]*len(subset))).astype(float)

        avg_impact_score.append(round(impact.mean(), 2))
        avg_model_confidence_score.append(round(conf.mean(), 2))
        avg_senti_score.append(round(senti.mean(), 2))

        # Growth %
        if "future_growth" in subset.columns:
            count_yes = (subset["future_growth"].astype(str).str.lower() == "yes").sum()
            total = subset["future_growth"].count()
        else:
            count_yes = 0
            total = 0
        percentage_yes = (count_yes / total) * 100 if total > 0 else 0
        percentage_yes = 100 if percentage_yes > 100 else percentage_yes
        growth_percent.append(percentage_yes)

        group_values.append(i)

    col_name = rename_col if rename_col else group_col
    final_df = pd.DataFrame({
        "avg_impact_score": avg_impact_score,
        "avg_model_confidence_score": [val * 100 for val in avg_model_confidence_score],
        "avg_senti_score": [val * 100 for val in avg_senti_score],
        "growth_percent": growth_percent,
        col_name: group_values
    }).sort_values(by="avg_model_confidence_score", ascending=True)

    fig = px.bar(
        final_df,
        y=col_name,
        x=["avg_impact_score", "avg_senti_score", "growth_percent", "avg_model_confidence_score"],
        orientation='h',
        barmode="group",
        title=title,
        height=700
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black')
    )
    fig.update_xaxes(showticklabels=True, ticks="outside", ticklen=5, tickwidth=2,
                     tickcolor='black', showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)

    return final_df, fig

# =======================
# 🧭 App Header
# =======================
st.title("📊 Stock & News Analysis Dashboard")
st.markdown("Bringing together stock data 📈 and market news 📰 for smarter decisions. Based on Newspapers and articles from various websites summarized by AI")

# Inject CSS for wrapping & margins
st.markdown(
    """
    <style>
      #stock-and-news-analysis-dashboard { margin-top: -2%; font-size: 30px; }
      h3 { font-size: 17px; margin-top: -10px; }
      h2 { font-size: 17px; margin-top: -10px; margin-bottom: -12px; }
      .stMainBlockContainer { margin-top: -1%; }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# 🗂️ Tabs
# =======================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📑 News Report", "🗃️ Data Tables"])

# -----------------------
# ✅ First Tab: Dashboard
# -----------------------
with tab1:
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365)

    # -----------------------------
    # Helper function: Technical Dashboard
    # -----------------------------
    def stock_dashboard(ticker, start=start_date, end=end_date):
        data = yf.download(ticker, start=start, end=end, group_by="column", auto_adjust=False)

        # Flatten MultiIndex if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No data for {ticker}")
            return fig, "HOLD ⏸️"

        # -----------------------------
        # Fibonacci Levels (last 10 days)
        # -----------------------------
        fib_data = data.tail(10)  # last 10 days
        high, low = fib_data['High'].max(), fib_data['Low'].min()
        diff = high - low if pd.notna(high) and pd.notna(low) else 0
        levels = {
            '0.0%': high if pd.notna(high) else 0,
            '23.6%': (high - 0.236 * diff) if diff else 0,
            '38.2%': (high - 0.382 * diff) if diff else 0,
            '50.0%': (high - 0.5 * diff) if diff else 0,
            '61.8%': (high - 0.618 * diff) if diff else 0,
            '100.0%': low if pd.notna(low) else 0
        }

        # -----------------------------
        # MACD
        # -----------------------------
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # -----------------------------
        # RSI (simple rolling)
        # -----------------------------
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # -----------------------------
        # Buy/Sell signals
        # -----------------------------
        buy_signals, sell_signals = [], []
        for i in range(1, len(data)):
            if macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1] and rsi.iloc[i] < 70:
                buy_signals.append((data.index[i], data['Close'].iloc[i]))
            elif macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1] and rsi.iloc[i] > 30:
                sell_signals.append((data.index[i], data['Close'].iloc[i]))

        # -----------------------------
        # Plotly Chart
        # -----------------------------
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f"{ticker} Price with Fibonacci Levels", "MACD", "RSI")
        )

        # Fibonacci shading for last 10 days
        fib_colors = ["rgba(255,87,51,0.2)", "rgba(255,195,0,0.2)", "rgba(218,247,166,0.2)",
                      "rgba(51,255,189,0.2)", "rgba(51,128,255,0.2)"]
        fib_names = ['0.0%', '23.6%', '38.2%', '50.0%', '61.8%', '100.0%']
        fib_dates = fib_data.index

        if len(fib_dates) > 0 and diff:
            for i in range(len(fib_names)-1):
                fig.add_trace(go.Scatter(
                    x=list(fib_dates) + list(fib_dates[::-1]),
                    y=[levels[fib_names[i]]]*len(fib_dates) + [levels[fib_names[i+1]]]*len(fib_dates),
                    fill='toself',
                    fillcolor=fib_colors[i % len(fib_colors)],
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)

        # Candlestick (on top of shaded area)
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Candlesticks",
            increasing_line_color="limegreen",
            decreasing_line_color="red"
        ), row=1, col=1)

        # Optional: add horizontal lines for Fibonacci reference
        for name, price in levels.items():
            if price:
                fig.add_hline(
                    y=price,
                    line=dict(color='white', dash="dot", width=1),
                    annotation_text=f"{name} {price:.2f}",
                    annotation_position="right",
                    annotation_font=dict(color='white', size=10),
                    row=1, col=1
                )

        # Buy/Sell signals
        if buy_signals:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in buy_signals],
                y=[p for _, p in buy_signals],
                mode="markers", name="Buy Signal",
                marker=dict(symbol="triangle-up", color="lime", size=12, line=dict(width=1, color="black"))
            ), row=1, col=1)
        if sell_signals:
            fig.add_trace(go.Scatter(
                x=[d for d, _ in sell_signals],
                y=[p for _, p in sell_signals],
                mode="markers", name="Sell Signal",
                marker=dict(symbol="triangle-down", color="crimson", size=12, line=dict(width=1, color="black"))
            ), row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=macd, mode="lines", name="MACD", line=dict(color="cyan", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=signal, mode="lines", name="Signal", line=dict(color="orange", width=2)), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=rsi, mode="lines", name="RSI", line=dict(color="violet", width=2)), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="limegreen", dash="dot"), row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=900,
            autosize=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=80, b=50),
            title=dict(text=f"{ticker} Technical Analysis Dashboard", font=dict(size=22, color="aqua")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Recommendation
        latest_macd, latest_signal, latest_rsi = macd.iloc[-1], signal.iloc[-1], rsi.iloc[-1]
        if latest_macd > latest_signal and latest_rsi < 70:
            recommendation = "BUY ✅"
        elif latest_macd < latest_signal and latest_rsi > 30:
            recommendation = "SELL ❌"
        else:
            recommendation = "HOLD ⏸️"

        return fig, recommendation

    st.markdown("---")

    # Input Section
    st.subheader("🔍 Select Stock(s)")
    stocks = st.text_input(
        "Enter stock symbols (comma separated, NSE/BSE tickers):",
        "ITC.NS, INFY.NS, RELIANCE.NS"
    )
    stock_list = [s.strip() for s in stocks.split(",") if s.strip()]
    selected_stock_dash = st.selectbox("Choose a stock to analyze", stock_list) if stock_list else None
    st.markdown("### ")

    # Dashboard Section
    if selected_stock_dash:
        fig_dash, reco = stock_dashboard(selected_stock_dash)
        st.plotly_chart(fig_dash, use_container_width=True)

        # Recommendation Section
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; 
                            background-color:#1E1E1E; text-align:center; 
                            border: 1px solid #444;">
                    <h3 style="color:aqua;">📊 Recommendation</h3>
                    <h2 style="color:lime;">{reco}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------
# ✅ Second Tab: News Report
# -----------------------
with tab2:
    # Candlestick chart below
    st.subheader("📈 Candlestick Chart")
    row1_c1_f1, row1_c1_f2, row1_c1_f3 = st.columns([1, 1, 1])
    with row1_c1_f1:
        nse_options = sorted(stock_df['NSE'].dropna().unique())
        selected_nse_tab2 = st.selectbox("Select NSE", nse_options)
    with row1_c1_f2:
        filtered_stock = stock_df[stock_df['NSE'] == selected_nse_tab2]
        stock_names = filtered_stock['STOCK_NAME'].dropna().unique()
        selected_stock_tab2 = st.selectbox("Select Stock", stock_names)

    stock_data = filtered_stock[filtered_stock['STOCK_NAME'] == selected_stock_tab2].sort_values("Date")

    row1_c1_c1, row1_c1_c2 = st.columns([4, 1])
    with row1_c1_c1:
        fig_candle = go.Figure(data=[go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        fig_candle.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

    row2_c1, row2_c2 = st.columns([1, 1])
    row3_c1, row3_c2 = st.columns([1, 1])

    with row2_c1:
        # Indian Market Overview first
        st.markdown("### 🌍 Indian Market Overview (Sector Group)")
        final_df_sector, fig_sector = analyze_and_plot(
            news_df,
            group_col="Sector Group",
            title="Indian Market - Overview",
            rename_col="Economy"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with row2_c2:
        # Stock Relative below
        st.markdown("### 📊 Stock Relative Analysis")
        final_df_relative, fig_relative = analyze_and_plot(
            news_df,
            group_col="Stock Relative",
            title="Scores by Stock Relative",
            rename_col="Stock_Relative"
        )
        st.plotly_chart(fig_relative, use_container_width=True)

    with row3_c1:
        # NSE first
        st.markdown("### 📊 Scores by Nifty (NSE)")
        final_df_nse, fig_nse = analyze_and_plot(
            news_df,
            group_col="NSE",
            title="Scores by Nifty",
            rename_col="NSE"
        )
        st.plotly_chart(fig_nse, use_container_width=True)

# -----------------------
# ✅ Third Tab: Data Tables
# -----------------------
with tab3:
    st.header("📰 News Data Explorer")

    # --- Ensure string type for categorical columns ---
    string_cols = ["NSE", "Stock Relative", "Sector Group", "future_growth", "type"]
    for col in string_cols:
        if col in news_df.columns:
            news_df[col] = news_df[col].fillna("Unknown").astype(str)

    # ---------- Defaults ----------
    DEFAULT_IMPACT = 90
    DEFAULT_SENTIMENT = 90
    DEFAULT_GROWTH = "Yes"

    # Helper: find default index in a list (case-insensitive), with "All" prepended
    def default_index(options_list, default_value):
        opts = ["All"] + options_list
        for i, v in enumerate(opts):
            if str(v).strip().lower() == str(default_value).strip().lower():
                return i
        return 0  # fallback to "All"

    # --- Base filtered df (will cascade with selections) ---
    filtered_news = news_df.copy()

    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)

    # DATE
    with col_f:
        if "date" in filtered_news.columns:
            date_series = pd.to_datetime(filtered_news["date"], errors="coerce")
            date_options = sorted(date_series.dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        else:
            date_options = []
        selected_date = st.selectbox("Select Date", ["All"] + date_options,
                                     index=len(["All"] + date_options) - 1 if date_options else 0)

    temp_df = filtered_news.copy()
    if selected_date != "All" and "date" in temp_df.columns:
        temp_df = temp_df[pd.to_datetime(temp_df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == selected_date]

    # TYPE (Sector/Commodity)
    with col_a:
        type_options = sorted(temp_df["type"].dropna().unique().tolist()) if "type" in temp_df.columns else []
        selected_type = st.selectbox("Select Sector/Commodity", ["All"] + type_options, index=0)
    if selected_type != "All" and "type" in temp_df.columns:
        temp_df = temp_df[temp_df["type"].str.strip().str.lower() == selected_type.strip().lower()]

    # STOCK RELATIVE
    with col_b:
        stock_options = sorted(temp_df["Stock Relative"].dropna().unique().tolist()) if "Stock Relative" in temp_df.columns else []
        selected_stock_rel = st.selectbox("Select Stock Relative", ["All"] + stock_options, index=0)
    if selected_stock_rel != "All" and "Stock Relative" in temp_df.columns:
        temp_df = temp_df[temp_df["Stock Relative"] == selected_stock_rel]

    # FUTURE GROWTH
    with col_c:
        growth_options = sorted(temp_df["future_growth"].dropna().unique().tolist()) if "future_growth" in temp_df.columns else []
        growth_idx = default_index(growth_options, DEFAULT_GROWTH)
        selected_growth = st.selectbox("Future Growth", ["All"] + growth_options,
                                       index=min(growth_idx, len(["All"] + growth_options) - 1))
    if selected_growth != "All" and "future_growth" in temp_df.columns:
        temp_df = temp_df[temp_df["future_growth"].str.strip().str.lower() == selected_growth.strip().lower()]

    # Impact / Sentiment thresholds (display + filter)
    with col_d:
        impact_threshold = st.number_input("Impact Score Threshold", min_value=0, max_value=100, value=DEFAULT_IMPACT)
    with col_e:
        sentiment_threshold = st.number_input("Sentiment Score Threshold", min_value=0, max_value=100, value=DEFAULT_SENTIMENT, step=1)

    # Apply numeric filters if columns exist (auto-scale 0–1 to 0–100 for comparison)
    def scaled_filter(series, threshold):
        s = pd.to_numeric(series, errors="coerce")
        smax = s.max(skipna=True)
        if pd.notna(smax) and smax <= 1:
            s = s * 100
        return s >= threshold  # keep items meeting or exceeding threshold

    if "impact_score" in temp_df.columns:
        mask_imp = scaled_filter(temp_df["impact_score"], impact_threshold)
        temp_df = temp_df[mask_imp.fillna(False)]

    if "sentiment_score" in temp_df.columns:
        mask_sen = scaled_filter(temp_df["sentiment_score"], sentiment_threshold)
        temp_df = temp_df[mask_sen.fillna(False)]

    # --- Normalize columns for display-only (Title-Case) ---
    temp_df_disp = temp_df.copy()
    temp_df_disp.columns = [c.strip().title() for c in temp_df_disp.columns]

    # --- Final dedupe by Headline + Summary if present ---
    subset_cols = [c for c in ["Headline", "Summary"] if c in temp_df_disp.columns]
    if subset_cols:
        temp_df_disp = temp_df_disp.drop_duplicates(subset=subset_cols)

    # Table columns (only if available)
    col_names = ["Headline", "Summary", "Sector_Or_Commodity", "Market_Sentiment",
                 "Sentiment_Score", "Impact_Score", "Confidence_Score", "Stock_Name", "Sector Group"]

    if not temp_df_disp.empty and all(c in temp_df_disp.columns for c in col_names):
        results_df = temp_df_disp.reset_index(drop=True)[col_names]
        results_df = results_df.sort_values(by="Impact_Score", ascending=False)
        results_df.index = results_df.index + 1

        gb = GridOptionsBuilder.from_dataframe(results_df)
        # Make first two columns wider & wrapped
        first_two_columns = results_df.columns[:2]
        gb.configure_column(first_two_columns[0], width=250, wrapText=True, autoHeight=True)
        gb.configure_column(first_two_columns[1], width=450, wrapText=True, autoHeight=True)
        gridOptions = gb.build()

        AgGrid(
            results_df,
            gridOptions=gridOptions,
            # columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW
        )

        st.caption(f"Showing {len(results_df)} articles after filtering")
    else:
        st.info("No news articles found for the applied filters.")
