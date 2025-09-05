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
st.set_page_config(layout="wide")  # ✅ Wide mode (no scrolling)

# -----------------------------
# Helper function: Technical Dashboard
# -----------------------------
def stock_dashboard(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, group_by="column", auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {ticker}")
        return fig, "HOLD ⏸️"

    # -----------------------------
    # Fibonacci (last 10 days)
    # -----------------------------
    fib_data = data.tail(10)
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
    # MACD (kept for recommendation only)
    # -----------------------------
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # -----------------------------
    # Plotly Chart (2 rows: Candles + RSI)
    # -----------------------------
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Candlestick + Fibonacci", "RSI")
    )

    # Fibonacci shading
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

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name="Candlestick",
        increasing_line_color="limegreen",
        decreasing_line_color="red"
    ), row=1, col=1)

    # Fibonacci horizontal lines
    for name, price in levels.items():
        if price:
            fig.add_hline(
                y=price,
                line=dict(color='white', dash="dot", width=1),
                annotation_text=f"{name} {price:.2f}",
                annotation_position="right",
                annotation_font=dict(color='white', size=9),
                row=1, col=1
            )

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=rsi, mode="lines", name="RSI",
                             line=dict(color="violet", width=2)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="limegreen", dash="dot"), row=2, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=650,  # ✅ more compact
        margin=dict(l=30, r=30, t=60, b=40),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{ticker} Technical Dashboard", font=dict(size=20, color="aqua")),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
    )

    # Zoom to last 6 weeks
    if len(data) > 30:
        fig.update_xaxes(range=[data.index[-30], data.index[-1]])

    # Recommendation
    latest_macd, latest_signal, latest_rsi = macd.iloc[-1], signal.iloc[-1], rsi.iloc[-1]
    if latest_macd > latest_signal and latest_rsi < 70:
        recommendation = "BUY ✅"
    elif latest_macd < latest_signal and latest_rsi > 30:
        recommendation = "SELL"
    else:
        recommendation = "HOLD ⏸️"

    return fig, recommendation


# -----------------------------
# Streamlit UI
# -----------------------------
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=365)

st.subheader("🔍 Stock Analysis Dashboard")

stocks = st.text_input("Enter stock symbols (comma separated):", "ITC.NS, INFY.NS, GAIL.NS, RELIANCE.NS, TMB.NS, HDFCBANK.NS, TCS.NS, HINDUNILVR.NS, ASIANPAINT.NS, SOUTHBANK.NS, TATAMOTORS.NS")
stock_list = [s.strip() for s in stocks.split(",") if s.strip()]
selected_stock_dash = st.selectbox("Choose a stock to analyze", stock_list) if stock_list else None

if selected_stock_dash:
    fig_dash, reco = stock_dashboard(selected_stock_dash, start_date, end_date)

    col_left, col_right = st.columns([4, 1])  # ✅ Chart wide, reco compact
    with col_left:
        st.plotly_chart(fig_dash, use_container_width=True)
    with col_right:
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; 
                        background-color:#1E1E1E; text-align:center; 
                        border: 1px solid #444; margin-top:50px;">
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

    # ---------- Defaults you asked for ----------
    DEFAULT_IMPACT = 90
    DEFAULT_SENTIMENT = 90
    # DEFAULT_DATE_STR = news_df["date"].max()#"2025-08-16"          # yyyy-mm-dd
    DEFAULT_GROWTH = "Yes"
    # DEFAULT_TYPE = "commodity"
    # DEFAULT_NSE = "Nifty Energy"
    # DEFAULT_STOCK_REL = "Energy & Resources"
    # --------------------------------------------

    # --- User inputs for thresholds (0–100 for both, we’ll auto-scale data if needed) ---


    # --- Apply numeric filters first (auto-scale series to 0–100 if needed) ---
    filtered_news = news_df.copy()

    # if "impact_score" in filtered_news.columns:
    #     imp = pd.to_numeric(filtered_news["impact_score"], errors="coerce")
    #     imp_max = imp.max(skipna=True)
    #     if pd.notna(imp_max) and imp_max <= 1:
    #         imp = imp * 100
    #     filtered_news = filtered_news[imp < impact_threshold]

    # if "sentiment_score" in filtered_news.columns:
    #     sen = pd.to_numeric(filtered_news["sentiment_score"], errors="coerce")
    #     sen_max = sen.max(skipna=True)
    #     if pd.notna(sen_max) and sen_max <= 1:
    #         sen = sen * 100
    #     filtered_news = filtered_news[sen < sentiment_threshold]

    # st.subheader("Dropdown Filters")

    # Helper: find default index in a list (case-insensitive), with "All" prepended
    def default_index(options_list, default_value):
        opts = ["All"] + options_list
        for i, v in enumerate(opts):
            if str(v).strip().lower() == str(default_value).strip().lower():
                return i
        return 0  # fallback to "All"

    # --- Dropdowns with cascading options ---
    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)

    # DATE
    with col_f:
        if "date" in filtered_news.columns:
            # Ensure consistent string format yyyy-mm-dd
            date_series = pd.to_datetime(filtered_news["date"], errors="coerce")
            date_options = sorted(date_series.dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        else:
            date_options = []
        # date_idx = default_index(date_options, DEFAULT_DATE_STR)
        selected_date = st.selectbox("Select Date", ["All"] + date_options, index=len(["All"] + date_options) - 1)

    temp_df = filtered_news.copy()
    if selected_date != "All" and "date" in temp_df.columns:
        temp_df = temp_df[pd.to_datetime(temp_df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == selected_date]


    # TYPE (Sector/Commodity)
    with col_a:
        type_options = sorted(temp_df["type"].dropna().unique().tolist()) if "type" in temp_df.columns else []
        # type_idx = default_index(type_options, DEFAULT_TYPE)
        selected_type = st.selectbox("Select Sector/Commodity", ["All"] + type_options, index=0)#min(type_idx, len(["All"] + type_options) - 1)
    if selected_type != "All" and "type" in temp_df.columns:
        temp_df = temp_df[temp_df["type"].str.strip().str.lower() == selected_type.strip().lower()]


    # NSE
    # with col_d:
    #     nse_options = sorted(temp_df["NSE"].dropna().unique().tolist()) if "NSE" in temp_df.columns else []
    #     nse_idx = default_index(nse_options, DEFAULT_NSE)
    #     selected_nse = st.selectbox("Select NSE", ["All"] + nse_options, index=min(nse_idx, len(["All"] + nse_options) - 1))
    # if selected_nse != "All" and "NSE" in temp_df.columns:
    #     temp_df = temp_df[temp_df["NSE"] == selected_nse]

    # STOCK RELATIVE
    with col_b:
        stock_options = sorted(temp_df["Stock Relative"].dropna().unique().tolist()) if "Stock Relative" in temp_df.columns else []
        # stock_idx = default_index(stock_options, DEFAULT_STOCK_REL)
        selected_stock = st.selectbox("Select Stock Relative", ["All"] + stock_options, index=0)
    if selected_stock != "All" and "Stock Relative" in temp_df.columns:
        temp_df = temp_df[temp_df["Stock Relative"] == selected_stock]

    # SECTOR GROUP
    # with col_b:
    #     sector_options = sorted(temp_df["Sector Group"].dropna().unique().tolist()) if "Sector Group" in temp_df.columns else []
    #     # No explicit default for Sector Group requested; default to "All"
    #     selected_sector = st.selectbox("Select Sector Group", ["All"] + sector_options)
    # if selected_sector != "All" and "Sector Group" in temp_df.columns:
    #     temp_df = temp_df[temp_df["Sector Group"] == selected_sector]


    # col1, col2,col3, col4 = st.columns(4)
    with col_d:
        impact_threshold = st.number_input(
            "Impact Score Threshold",
            min_value=0, max_value=100, value=DEFAULT_IMPACT
        )
    with col_e:
        sentiment_threshold = st.number_input(
            "Sentiment Score Threshold",
            min_value=0, max_value=100, value=DEFAULT_SENTIMENT, step=1
        )
    # FUTURE GROWTH
    with col_c:
        growth_options = sorted(temp_df["future_growth"].dropna().unique().tolist()) if "future_growth" in temp_df.columns else []
        growth_idx = default_index(growth_options, DEFAULT_GROWTH)
        selected_growth = st.selectbox("Future Growth", ["All"] + growth_options, index=min(growth_idx, len(["All"] + growth_options) - 1))
        if selected_growth != "All" and "future_growth" in temp_df.columns:
            temp_df = temp_df[temp_df["future_growth"].str.strip().str.lower() == selected_growth.strip().lower()]



    # --- Normalize columns for display-only (Title-Case) ---
    temp_df_disp = temp_df.copy()
    temp_df_disp.columns = [c.strip().title() for c in temp_df_disp.columns]

    # --- Final Filtered News (dedupe by Headline + Summary if they exist) ---
    subset_cols = [c for c in ["Headline", "Summary"] if c in temp_df_disp.columns]
    if subset_cols:
        temp_df_disp = temp_df_disp.drop_duplicates(subset=subset_cols)



    # st.subheader("Filtered News Results")
    col_names=["Headline", "Summary","Sector_Or_Commodity","Market_Sentiment","Sentiment_Score","Impact_Score","Confidence_Score","Stock_Name","Sector Group"]
    if not temp_df_disp.empty and all(c in temp_df_disp.columns for c in col_names):
        results_df = temp_df_disp.reset_index(drop=True)[col_names]
        results_df = results_df.sort_values(by="Impact_Score", ascending=False)
        results_df.index = results_df.index + 1
#        results_df = results_df.reset_index().rename(columns={"index": "S.No"})

        gb = GridOptionsBuilder.from_dataframe(results_df)
        # gb.configure_default_column(resizable=True)
        # gb.configure_column(field="Headline",header_name="Headline", width=300, cellStyle={"white-space": "normal", "word-wrap": "break-word"})
        first_two_columns = results_df.columns[:2]
        gb.configure_column(first_two_columns[0], width=100)
        gb.configure_column(first_two_columns[1],width=100,
    cellStyle={"white-space": "normal", "word-wrap": "break-word"},
    wrapText=True,
    autoHeight=True
)
        gridOptions = gb.build()

        AgGrid(results_df, gridOptions=gridOptions
               # , columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW
               )

        # AgGrid(results_df,gridOptions=gridOptions,columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

        # Inject CSS for wrapping & narrow S.No column
        # st.markdown(
        #     """
        #     <style>
        #       table { table-layout: fixed; width: 100%; }
        #       th, td { white-space: normal !important; word-wrap: break-word !important; }
        #       th:nth-child(1), td:nth-child(1) { width: 60px; }      /* S.No */
        #       th:nth-child(2), td:nth-child(2) { width: 35%; }       /* Headline */
        #       th:nth-child(3), td:nth-child(3) { width: 55%; }       /* Summary */
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )
        #st.markdown(html_table, unsafe_allow_html=True)
        # components.html(html_code, height=700, scrolling=True)

        st.caption(f"Showing {len(results_df)} articles after filtering")
    else:
        st.info("No news articles found for the applied filters.")
