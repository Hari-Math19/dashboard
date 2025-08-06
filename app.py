import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- Load Data ---
data_dir = os.path.join(os.getcwd(), 'data')
np_file = os.path.join(data_dir, 'merged_output_np.xlsx')
stocks_file = os.path.join(data_dir, 'merged_output_stocks.xlsx')

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        st.error(f"Failed to load {file_path}: {e}")
        return pd.DataFrame()

df_news = load_data(np_file)
df_stocks = load_data(stocks_file)

# --- App Config ---
st.set_page_config("📊 News & Stock Dashboard", layout="wide")
st.title("📊 Interactive News & Stock Dashboard")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📰 News Data", "📈 Stock Data", "📊 Pivot Analysis"])

# --- Tab 1: News Data ---
with tab1:
    st.header("📰 News Data Dashboard")

    categories = st.multiselect("Select Category", sorted(df_news['category'].dropna().unique()) if 'category' in df_news.columns else [])
    sectors = st.multiselect("Select Sector", sorted(df_news['sector'].dropna().unique()) if 'sector' in df_news.columns else [])
    filtered_news = df_news.copy()
    if categories and 'category' in filtered_news.columns:
        filtered_news = filtered_news[filtered_news['category'].isin(categories)]
    if sectors and 'sector' in filtered_news.columns:
        filtered_news = filtered_news[filtered_news['sector'].isin(sectors)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Market Sentiment Distribution")
        if 'market_sentiment' in filtered_news.columns:
            sentiment_count = filtered_news['market_sentiment'].value_counts().reset_index()
            sentiment_count.columns = ['Sentiment', 'Count']
            fig = px.pie(sentiment_count, names='Sentiment', values='Count', title='Sentiment Breakdown')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 Avg Sentiment Score by Sector")
        if 'sentiment_score' in filtered_news.columns and 'sector' in filtered_news.columns:
            avg_scores = filtered_news.groupby('sector')['sentiment_score'].mean().reset_index()
            fig = px.bar(avg_scores, x='sector', y='sentiment_score', title='Avg Sentiment Score by Sector')
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗞️ Filtered News Table")
    st.dataframe(filtered_news, use_container_width=True)

# --- Tab 2: Stock Data ---
with tab2:
    st.header("📈 Stock Data Dashboard")

    # Filter: Sector
    if 'sector' in df_stocks.columns:
        stock_sectors = st.multiselect("Select Sector", sorted(df_stocks['sector'].dropna().unique()))
    else:
        stock_sectors = []

    # Filter: Stock Name
    if 'stock_name' in df_stocks.columns:
        stock_names = st.multiselect("Select Stock Name", sorted(df_stocks['stock_name'].dropna().unique()))
    else:
        stock_names = []

    # Apply filters
    filtered_stocks = df_stocks.copy()
    if stock_sectors:
        filtered_stocks = filtered_stocks[filtered_stocks['sector'].isin(stock_sectors)]
    if stock_names:
        filtered_stocks = filtered_stocks[filtered_stocks['stock_name'].isin(stock_names)]

    # Optional Date filter
    if 'date' in df_stocks.columns:
        df_stocks['date'] = pd.to_datetime(df_stocks['date'], errors='coerce')
        min_date = df_stocks['date'].min()
        max_date = df_stocks['date'].max()
        date_range = st.date_input("Select Date Range", value=(min_date, max_date))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_stocks = filtered_stocks[(filtered_stocks['date'] >= start) & (filtered_stocks['date'] <= end)]

    # Show pivot table area
    st.subheader("📊 Pivot Table Builder")
    pivot_cols = st.multiselect("Select columns to include in Pivot Table", filtered_stocks.columns.tolist())
    if pivot_cols:
        st.dataframe(filtered_stocks[pivot_cols], use_container_width=True)

    # Multicolumn plot
    st.subheader("📈 Multicolumn Plot")
    numeric_cols = filtered_stocks.select_dtypes(include='number').columns.tolist()
    selected_y = st.multiselect("Select numeric columns to plot", numeric_cols)
    if selected_y:
        fig = px.line(filtered_stocks, x='date' if 'date' in filtered_stocks.columns else filtered_stocks.index, y=selected_y)
        st.plotly_chart(fig, use_container_width=True)

    # Show raw filtered table
    st.subheader("📋 Filtered Stock Table")
    st.dataframe(filtered_stocks, use_container_width=True)

# --- Tab 3: Pivot Analysis ---
with tab3:
    st.header("📊 Pivot Table Workspace")

    dataset_choice = st.selectbox("Select Dataset", ["News", "Stocks"])
    df_source = df_news if dataset_choice == "News" else df_stocks

    if df_source.empty:
        st.warning("Selected dataset is empty.")
    else:
        numeric_columns = df_source.select_dtypes(include='number').columns.tolist()
        all_columns = df_source.columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            index = st.selectbox("Index (Rows)", options=all_columns)
        with col2:
            columns = st.selectbox("Columns (Optional)", options=[None] + all_columns)
        with col3:
            values = st.selectbox("Values (Agg Target)", options=numeric_columns)

        aggfunc = st.selectbox("Aggregation Function", options=['sum', 'mean', 'count', 'max', 'min'])

        if st.button("Generate Pivot Table"):
            try:
                pivot_df = pd.pivot_table(
                    df_source,
                    index=index,
                    columns=columns if columns else None,
                    values=values,
                    aggfunc=aggfunc,
                    fill_value=0
                )

                st.subheader("📋 Pivot Table Result")
                st.dataframe(pivot_df, use_container_width=True)

                st.subheader("📊 Visualization")
                chart_type = st.radio("Chart Type", ['Bar Chart', 'Line Chart'])

                # Multi-column plotting
                if columns:
                    pivot_df_plot = pivot_df.reset_index()
                    pivot_df_plot.columns.name = None
                    fig = px.bar(pivot_df_plot, x=index, y=pivot_df_plot.columns[1:]) if chart_type == "Bar Chart" \
                        else px.line(pivot_df_plot, x=index, y=pivot_df_plot.columns[1:])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    pivot_df_plot = pivot_df.reset_index()
                    fig = px.bar(pivot_df_plot, x=index, y=values) if chart_type == "Bar Chart" \
                        else px.line(pivot_df_plot, x=index, y=values)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating pivot: {e}")
