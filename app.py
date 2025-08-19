import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =======================
# 📂 Load Data
# =======================
@st.cache_data
def load_data():
    stock_df = pd.read_csv(r"data\merged_output_17_08_2025.csv", parse_dates=["Date"])
    news_df = pd.read_csv(r"data\combined_output.csv", parse_dates=["date"])
    # Replace NaN with 0
    stock_df = stock_df.fillna(0)
    news_df = news_df.fillna(0)
    return stock_df, news_df

stock_df, news_df = load_data()

# =======================
# 📊 Analysis Function
# =======================
def analyze_and_plot(df, group_col, title, rename_col=None):
    avg_impact_score = []
    avg_model_confidence_score = []
    avg_senti_score = []
    group_values = []
    growth_percent = []

    for i in df[group_col].dropna().unique():
        subset = df[df[group_col] == i]

        avg_impact_score.append(round(subset['impact_score'].mean(), 2))
        avg_model_confidence_score.append(round(subset['confidence_score'].mean(), 2))
        avg_senti_score.append(round(subset['sentiment_score'].mean(), 2))

        # Growth %
        count_yes = (subset["future_growth"] == "yes").sum()
        total = subset["future_growth"].count()
        percentage_yes = (count_yes / total) * 100 if total > 0 else 0
        growth_percent.append(percentage_yes)

        group_values.append(i)

    col_name = rename_col if rename_col else group_col
    final_df = pd.DataFrame({
        "avg_impact_score": avg_impact_score,
        "avg_model_confidence_score": [val * 100 for val in avg_model_confidence_score],
        "avg_senti_score": [val * 100 for val in avg_senti_score],
        "growth_percent": growth_percent,
        col_name: group_values
    })

    fig = px.bar(
        final_df,
        x=col_name,
        y=["avg_impact_score", "avg_senti_score", "growth_percent", "avg_model_confidence_score"],
        barmode="group",
        title=title
    )

    return final_df, fig


# =======================
# 📌 Streamlit Layout
# =======================
st.set_page_config(layout="wide")
st.title("📊 Stock & News Analysis Dashboard")
st.markdown("### Bringing together stock data 📈 and market news 📰 for smarter decisions.")

tab1, tab2 = st.tabs(["📊 Dashboard", "📑 News Report"])

# -----------------------
# ✅ First Tab: Dashboard
# -----------------------
with tab1:
    col_left, col_right = st.columns([1, 1])

    # Left Panel
    with col_left:
        # Indian Market Overview first
        st.markdown("### 🌍 Indian Market Overview (Sector Group)")
        final_df_sector, fig_sector = analyze_and_plot(
            news_df,
            group_col="Sector Group",
            title="Indian Market - Overview",
            rename_col="Economy"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

        # Candlestick chart below
        st.markdown("### 📈 Candlestick Chart")
        nse_options = sorted(stock_df['NSE'].dropna().unique())
        selected_nse = st.selectbox("Select NSE", nse_options)

        filtered_stock = stock_df[stock_df['NSE'] == selected_nse]
        stock_names = filtered_stock['STOCK_NAME'].dropna().unique()
        selected_stock = st.selectbox("Select Stock", stock_names)

        stock_data = filtered_stock[filtered_stock['STOCK_NAME'] == selected_stock].sort_values("Date")

        fig_candle = go.Figure(data=[go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        fig_candle.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)
    # Right Panel
    with col_right:
        # NSE first
        st.markdown("### 📊 Scores by Nifty (NSE)")
        final_df_nse, fig_nse = analyze_and_plot(
            news_df,
            group_col="NSE",
            title="Scores by Nifty",
            rename_col="NSE"
        )
        st.plotly_chart(fig_nse, use_container_width=True)

        # Stock Relative below
        st.markdown("### 📊 Stock Relative Analysis")
        final_df_relative, fig_relative = analyze_and_plot(
            news_df,
            group_col="Stock Relative",
            title="Scores by Stock Relative",
            rename_col="Stock_Relative"
        )
        st.plotly_chart(fig_relative, use_container_width=True)


# -----------------------
# ✅ Second Tab: Data Tables
# -----------------------
# --- Tab 2 Content (News Data Filtering) ---

with tab2:
    st.header("📰 News Data Explorer")

    # --- Ensure string type for categorical columns ---
    string_cols = ["NSE", "Stock Relative", "Sector Group", "future_growth", "type"]
    for col in string_cols:
        if col in news_df.columns:
            news_df[col] = news_df[col].fillna("Unknown").astype(str)

    # ---------- Defaults you asked for ----------
    DEFAULT_IMPACT = 90
    DEFAULT_SENTIMENT = 90
    DEFAULT_DATE_STR = "2025-08-16"          # yyyy-mm-dd
    DEFAULT_GROWTH = "Yes"
    DEFAULT_TYPE = "commodity"
    DEFAULT_NSE = "Nifty Energy"
    DEFAULT_STOCK_REL = "Energy & Resources"
    # --------------------------------------------

    # --- User inputs for thresholds (0–100 for both, we’ll auto-scale data if needed) ---
    col1, col2 = st.columns(2)
    with col1:
        impact_threshold = st.number_input(
            "Enter Impact Score Threshold (less than, 0–100)",
            min_value=0, max_value=100, value=DEFAULT_IMPACT
        )
    with col2:
        sentiment_threshold = st.number_input(
            "Enter Sentiment Score Threshold (less than, 0–100)",
            min_value=0, max_value=100, value=DEFAULT_SENTIMENT, step=1
        )

    # --- Apply numeric filters first (auto-scale series to 0–100 if needed) ---
    filtered_news = news_df.copy()

    if "impact_score" in filtered_news.columns:
        imp = pd.to_numeric(filtered_news["impact_score"], errors="coerce")
        imp_max = imp.max(skipna=True)
        if pd.notna(imp_max) and imp_max <= 1:
            imp = imp * 100
        filtered_news = filtered_news[imp < impact_threshold]

    if "sentiment_score" in filtered_news.columns:
        sen = pd.to_numeric(filtered_news["sentiment_score"], errors="coerce")
        sen_max = sen.max(skipna=True)
        if pd.notna(sen_max) and sen_max <= 1:
            sen = sen * 100
        filtered_news = filtered_news[sen < sentiment_threshold]

    st.subheader("Dropdown Filters")

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
    with col_a:
        if "date" in filtered_news.columns:
            # Ensure consistent string format yyyy-mm-dd
            date_series = pd.to_datetime(filtered_news["date"], errors="coerce")
            date_options = sorted(date_series.dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        else:
            date_options = []
        date_idx = default_index(date_options, DEFAULT_DATE_STR)
        selected_date = st.selectbox("Select Date", ["All"] + date_options, index=min(date_idx, len(["All"] + date_options) - 1))

    temp_df = filtered_news.copy()
    if selected_date != "All" and "date" in temp_df.columns:
        temp_df = temp_df[pd.to_datetime(temp_df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == selected_date]

    # FUTURE GROWTH
    with col_b:
        growth_options = sorted(temp_df["future_growth"].dropna().unique().tolist()) if "future_growth" in temp_df.columns else []
        growth_idx = default_index(growth_options, DEFAULT_GROWTH)
        selected_growth = st.selectbox("Future Growth", ["All"] + growth_options, index=min(growth_idx, len(["All"] + growth_options) - 1))
    if selected_growth != "All" and "future_growth" in temp_df.columns:
        temp_df = temp_df[temp_df["future_growth"].str.strip().str.lower() == selected_growth.strip().lower()]

    # TYPE (Sector/Commodity)
    with col_c:
        type_options = sorted(temp_df["type"].dropna().unique().tolist()) if "type" in temp_df.columns else []
        type_idx = default_index(type_options, DEFAULT_TYPE)
        selected_type = st.selectbox("Select Sector/Commodity", ["All"] + type_options, index=min(type_idx, len(["All"] + type_options) - 1))
    if selected_type != "All" and "type" in temp_df.columns:
        temp_df = temp_df[temp_df["type"].str.strip().str.lower() == selected_type.strip().lower()]

    # NSE
    with col_d:
        nse_options = sorted(temp_df["NSE"].dropna().unique().tolist()) if "NSE" in temp_df.columns else []
        nse_idx = default_index(nse_options, DEFAULT_NSE)
        selected_nse = st.selectbox("Select NSE", ["All"] + nse_options, index=min(nse_idx, len(["All"] + nse_options) - 1))
    if selected_nse != "All" and "NSE" in temp_df.columns:
        temp_df = temp_df[temp_df["NSE"] == selected_nse]

    # STOCK RELATIVE
    with col_e:
        stock_options = sorted(temp_df["Stock Relative"].dropna().unique().tolist()) if "Stock Relative" in temp_df.columns else []
        stock_idx = default_index(stock_options, DEFAULT_STOCK_REL)
        selected_stock = st.selectbox("Select Stock Relative", ["All"] + stock_options, index=min(stock_idx, len(["All"] + stock_options) - 1))
    if selected_stock != "All" and "Stock Relative" in temp_df.columns:
        temp_df = temp_df[temp_df["Stock Relative"] == selected_stock]

    # SECTOR GROUP
    with col_f:
        sector_options = sorted(temp_df["Sector Group"].dropna().unique().tolist()) if "Sector Group" in temp_df.columns else []
        # No explicit default for Sector Group requested; default to "All"
        selected_sector = st.selectbox("Select Sector Group", ["All"] + sector_options)
    if selected_sector != "All" and "Sector Group" in temp_df.columns:
        temp_df = temp_df[temp_df["Sector Group"] == selected_sector]

    # --- Normalize columns for display-only (Title-Case) ---
    temp_df_disp = temp_df.copy()
    temp_df_disp.columns = [c.strip().title() for c in temp_df_disp.columns]

    # --- Final Filtered News (dedupe by Headline + Summary if they exist) ---
    subset_cols = [c for c in ["Headline", "Summary"] if c in temp_df_disp.columns]
    if subset_cols:
        temp_df_disp = temp_df_disp.drop_duplicates(subset=subset_cols)

    st.subheader("Filtered News Results")

    if not temp_df_disp.empty and all(c in temp_df_disp.columns for c in ["Headline", "Summary"]):
        results_df = temp_df_disp.reset_index(drop=True)[["Headline", "Summary"]]
        results_df.index = results_df.index + 1
        results_df = results_df.reset_index().rename(columns={"index": "S.No"})

        # Render via HTML to control wrapping & S.No width
        # (st.dataframe truncates long text & wrapping is unreliable)
        html_table = results_df.to_html(escape=False, index=False)

        # Inject CSS for wrapping & narrow S.No column
        st.markdown(
            """
            <style>
              table { table-layout: fixed; width: 100%; }
              th, td { white-space: normal !important; word-wrap: break-word !important; }
              th:nth-child(1), td:nth-child(1) { width: 60px; }      /* S.No */
              th:nth-child(2), td:nth-child(2) { width: 35%; }       /* Headline */
              th:nth-child(3), td:nth-child(3) { width: 55%; }       /* Summary */
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(html_table, unsafe_allow_html=True)

        st.caption(f"Showing {len(results_df)} articles after filtering")
    else:
        st.info("No news articles found for the applied filters.")
