from pathlib import Path

import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

DATA_PATH = Path(__file__).parent / "GlobalWeatherRepository.csv"
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "GlobalWeatherRepository.csv"
MODEL_DIR = BASE_DIR / "notebooks" / "models"

XGB_MODEL = MODEL_DIR / "xgb_model.pkl"
RF_MODEL = MODEL_DIR / "rf_model.pkl"
MODEL_RESULTS = MODEL_DIR / "model_results.csv"
FEATURE_IMPORTANCE = MODEL_DIR / "feature_importance.csv"
MODEL_FEATURES = MODEL_DIR / "model_features.pkl"
PROCESSED_DATA = MODEL_DIR / "processed_weather_data.csv"
ISO_MODEL = MODEL_DIR / "isolation_forest.pkl"


@st.cache_resource
def load_models():

    xgb = joblib.load(XGB_MODEL)
    rf = joblib.load(RF_MODEL)

    return xgb, rf

@st.cache_resource
def load_models():

    xgb = joblib.load(XGB_MODEL)

    rf = joblib.load(RF_MODEL)

    iso = joblib.load(ISO_MODEL)

    return xgb, rf, iso

@st.cache_data
def load_results():

    results = pd.read_csv(MODEL_RESULTS)

    importance = pd.read_csv(FEATURE_IMPORTANCE)

    return results, importance

@st.cache_data
def load_data():

    return pd.read_csv(
        PROCESSED_DATA,
        parse_dates=["last_updated"]
    )


st.set_page_config(
    page_title="Weather Trend Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:1rem;
padding-bottom:2rem;
padding-left:2rem;
padding-right:2rem;
}

div[data-testid="stMetric"]{
background:#ffffff;
padding:18px;
border-radius:10px;
border:1px solid #e6e6e6;
box-shadow:0 1px 8px rgba(0,0,0,.05);
}

hr{
margin-top:0.3rem;
margin-bottom:0.8rem;
}

</style>
""", unsafe_allow_html=True)
def kpi_card(title, value):
    st.markdown(
        f"""
        <div style="
            background:#F8FAFC;
            border:1px solid #E5E7EB;
            border-radius:12px;
            padding:20px;
            text-align:center;
        ">
            <p style="margin:0;font-size:14px;color:#64748B;">{title}</p>
            <h2 style="margin:0;color:#111827;">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

xgb_model, rf_model, iso_model = load_models()

results_df, importance_df = load_results()

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["last_updated"] = pd.to_datetime(df["last_updated"])

    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    df["day"] = df["last_updated"].dt.day
    df["hour"] = df["last_updated"].dt.hour
    df["day_of_week"] = df["last_updated"].dt.dayofweek
    df["weekofyear"] = (
        df["last_updated"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn"
    }

    df["season"] = df["month"].map(season_map)

    return df


df = load_data()
filtered_df = df.copy()


st.sidebar.title("Weather Trend Forecasting")

st.sidebar.caption("PM Accelerator Technical Assessment")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Cleaning",
        "EDA",
        "Machine Learning",
        "Climate Analysis",
        "Air Quality",
        "Spatial Analysis",
        "Anomaly Detection",
        "About"
    ]
)

st.sidebar.divider()

countries = ["All"] + sorted(df["country"].unique())

selected_country = st.sidebar.selectbox(
    "Country",
    countries
)

if selected_country == "All":

    cities = ["All"] + sorted(df["location_name"].unique())

else:

    cities = ["All"] + sorted(
        df.loc[
            df["country"] == selected_country,
            "location_name"
        ].unique()
    )

selected_city = st.sidebar.selectbox(
    "City",
    cities
)

years = ["All"] + sorted(df["year"].unique().tolist())

selected_year = st.sidebar.selectbox(
    "Year",
    years
)

seasons = ["All"] + sorted(df["season"].unique())

selected_season = st.sidebar.selectbox(
    "Season",
    seasons
)

conditions = ["All"] + sorted(df["condition_text"].unique())

selected_condition = st.sidebar.selectbox(
    "Weather Condition",
    conditions
)

filtered_df = df.copy()

if selected_country != "All":

    filtered_df = filtered_df[
        filtered_df["country"] == selected_country
    ]

if selected_city != "All":

    filtered_df = filtered_df[
        filtered_df["location_name"] == selected_city
    ]

if selected_year != "All":

    filtered_df = filtered_df[
        filtered_df["year"] == selected_year
    ]

if selected_season != "All":

    filtered_df = filtered_df[
        filtered_df["season"] == selected_season
    ]

if selected_condition != "All":

    filtered_df = filtered_df[
        filtered_df["condition_text"] == selected_condition
    ]

st.sidebar.divider()

st.sidebar.subheader("Dataset")

st.sidebar.write(f"Rows : {len(filtered_df):,}")
st.sidebar.write(f"Countries : {filtered_df['country'].nunique()}")
st.sidebar.write(f"Cities : {filtered_df['location_name'].nunique()}")

st.sidebar.divider()


def dashboard():

    st.title("Weather Trend Forecasting Dashboard")
    st.caption("Interactive overview of the filtered weather dataset.")

    overview_tab, weather_tab, trends_tab, performance_tab, dataset_tab = st.tabs(
        [
            "Overview",
            "Weather",
            "Trends",
            "Performance",
            "Dataset"
        ]
    )

    with overview_tab:

        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)

        with kpi_1:
            kpi_card("Records", f"{len(filtered_df):,}")

        with kpi_2:
            kpi_card("Countries", filtered_df["country"].nunique())

        with kpi_3:
            kpi_card("Cities", filtered_df["location_name"].nunique())

        with kpi_4:
            kpi_card("Features", filtered_df.shape[1])

        st.divider()

        weather_1, weather_2, weather_3, weather_4 = st.columns(4)

        metrics = [
            ("Avg Temperature (°C)", filtered_df["temperature_celsius"].mean()),
            ("Avg Humidity (%)", filtered_df["humidity"].mean()),
            ("Avg Wind (kph)", filtered_df["wind_kph"].mean()),
            ("Avg Pressure (mb)", filtered_df["pressure_mb"].mean())
        ]

        for col, (title, value) in zip(
            [weather_1, weather_2, weather_3, weather_4],
            metrics
        ):
            with col:
                kpi_card(title, round(value, 2))

        st.divider()

        left, right = st.columns([2, 1])

        with left:

            monthly = (
                filtered_df
                .groupby("month")["temperature_celsius"]
                .mean()
                .reset_index()
            )

            fig = px.line(
                monthly,
                x="month",
                y="temperature_celsius",
                markers=True,
                template="plotly_white",
                title="Average Monthly Temperature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            seasonal = (
                filtered_df
                .groupby("season")["temperature_celsius"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                seasonal,
                x="season",
                y="temperature_celsius",
                color="season",
                template="plotly_white",
                title="Seasonal Temperature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with weather_tab:

        left, right = st.columns(2)

        with left:

            fig = px.histogram(
                filtered_df,
                x="temperature_celsius",
                nbins=40,
                template="plotly_white",
                title="Temperature Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            weather = (
                filtered_df["condition_text"]
                .value_counts()
                .head(10)
                .reset_index()
            )

            weather.columns = ["Condition", "Count"]

            fig = px.pie(
                weather,
                names="Condition",
                values="Count",
                hole=0.55,
                template="plotly_white",
                title="Weather Conditions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        fig = px.scatter(
            filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            ),
            x="humidity",
            y="temperature_celsius",
            color="wind_kph",
            template="plotly_white",
            title="Humidity vs Temperature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with trends_tab:

        left, right = st.columns(2)

        with left:

            yearly = (
                filtered_df
                .groupby("year")["temperature_celsius"]
                .mean()
                .reset_index()
            )

            fig = px.line(
                yearly,
                x="year",
                y="temperature_celsius",
                markers=True,
                template="plotly_white",
                title="Yearly Temperature Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            monthly = (
                filtered_df
                .groupby("month")["temperature_celsius"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                monthly,
                x="month",
                y="temperature_celsius",
                template="plotly_white",
                title="Monthly Temperature Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with performance_tab:

        best_model = (
            results_df
            .sort_values("R2", ascending=False)
            .iloc[0]
        )

        st.success(
            f"Best Model: {best_model['Model']} | R² = {best_model['R2']:.3f}"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        if "Feature" in importance_df.columns:

            fig = px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                template="plotly_white",
                title="Top 15 Feature Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with dataset_tab:

        columns = [
            "country",
            "location_name",
            "last_updated",
            "temperature_celsius",
            "humidity",
            "pressure_mb",
            "wind_kph",
            "condition_text"
        ]

        st.dataframe(
            filtered_df[columns],
            use_container_width=True,
            hide_index=True
        )

        csv = filtered_df.to_csv(index=False)

        st.download_button(
            "Download Filtered Dataset",
            csv,
            "filtered_weather_data.csv",
            "text/csv"
        )

        with st.expander("Summary Statistics"):

            st.dataframe(
                filtered_df.describe(),
                use_container_width=True
            )

def data_cleaning():

    st.title("Data Cleaning")
    st.caption("Overview of preprocessing steps applied before exploratory analysis and modeling.")

    overview_tab, missing_tab, duplicate_tab, datatype_tab, feature_tab = st.tabs(
        [
            "Overview",
            "Missing Values",
            "Duplicates",
            "Data Types",
            "Feature Engineering"
        ]
    )

    with overview_tab:

        duplicate_count = df.duplicated().sum()
        missing_count = df.isna().sum().sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            kpi_card("Rows", f"{len(df):,}")

        with col2:
            kpi_card("Columns", df.shape[1])

        with col3:
            kpi_card("Missing Values", missing_count)

        with col4:
            kpi_card("Duplicate Rows", duplicate_count)

        st.divider()

        st.subheader("Preprocessing Pipeline")

        pipeline = pd.DataFrame(
            {
                "Step": [
                    "Removed Duplicate Records",
                    "Handled Missing Values",
                    "Converted Date Columns",
                    "Extracted Date Features",
                    "Created Season Feature",
                    "Encoded Categorical Variables",
                    "Prepared Dataset for Machine Learning"
                ],
                "Status": ["Completed"] * 7
            }
        )

        st.dataframe(
            pipeline,
            use_container_width=True,
            hide_index=True
        )

    with missing_tab:

        missing = (
            df.isna()
            .sum()
            .reset_index()
        )

        missing.columns = [
            "Feature",
            "Missing Values"
        ]

        missing = missing.sort_values(
            "Missing Values",
            ascending=False
        )

        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            missing,
            x="Feature",
            y="Missing Values",
            template="plotly_white",
            title="Missing Values by Feature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with duplicate_tab:

        duplicate_count = df.duplicated().sum()

        kpi_card("Duplicate Records", duplicate_count)

        if duplicate_count > 0:

            st.dataframe(
                df[df.duplicated()].head(20),
                use_container_width=True
            )

        else:

            st.success("No duplicate records were found in the dataset.")

    with datatype_tab:

        dtype_df = pd.DataFrame(
            {
                "Feature": df.columns,
                "Data Type": df.dtypes.astype(str)
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True
        )

        dtype_count = (
            dtype_df["Data Type"]
            .value_counts()
            .reset_index()
        )

        dtype_count.columns = [
            "Data Type",
            "Count"
        ]

        fig = px.pie(
            dtype_count,
            names="Data Type",
            values="Count",
            hole=0.55,
            template="plotly_white",
            title="Data Type Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with feature_tab:

        engineered = pd.DataFrame(
            {
                "Feature": [
                    "year",
                    "month",
                    "day",
                    "hour",
                    "day_of_week",
                    "weekofyear",
                    "season"
                ],
                "Description": [
                    "Year extracted from timestamp",
                    "Month extracted from timestamp",
                    "Day extracted from timestamp",
                    "Hour extracted from timestamp",
                    "Weekday extracted from timestamp",
                    "ISO week number",
                    "Season derived from month"
                ]
            }
        )
        st.dataframe(
            engineered,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Feature Preview")
        preview_columns = [
            "last_updated",
            "year",
            "month",
            "day",
            "hour",
            "day_of_week",
            "weekofyear",
            "season"
        ]
        st.dataframe(
            df[preview_columns].head(20),
            use_container_width=True,
            hide_index=True
        )


def eda():
    st.title("Exploratory Data Analysis")
    st.caption("Interactive exploration of weather patterns, feature distributions, and relationships.")
    overview_tab, dist_tab, relation_tab, corr_tab, time_tab, insight_tab = st.tabs(
        [
            "Overview",
            "Distributions",
            "Relationships",
            "Correlations",
            "Time Analysis",
            "Insights"
        ]
    )
    with overview_tab:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            kpi_card(
                "Avg Temperature",
                f"{filtered_df['temperature_celsius'].mean():.2f} °C"
            )

        with col2:
            kpi_card(
                "Avg Humidity",
                f"{filtered_df['humidity'].mean():.2f} %"
            )

        with col3:
            kpi_card(
                "Avg Pressure",
                f"{filtered_df['pressure_mb'].mean():.2f}"
            )

        with col4:
            kpi_card(
                "Avg Wind",
                f"{filtered_df['wind_kph'].mean():.2f} kph"
            )

        st.divider()

        st.dataframe(
            filtered_df.describe(),
            use_container_width=True
        )

    with dist_tab:

        feature = st.selectbox(
            "Select Feature",
            [
                "temperature_celsius",
                "humidity",
                "pressure_mb",
                "wind_kph",
                "visibility_km",
                "precip_mm",
                "uv_index"
            ]
        )

        fig = px.histogram(
            filtered_df,
            x=feature,
            nbins=40,
            marginal="box",
            template="plotly_white",
            title=f"{feature.replace('_', ' ').title()} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with relation_tab:
        left, right = st.columns(2)
        numerical_features = [
            "temperature_celsius",
            "humidity",
            "pressure_mb",
            "wind_kph",
            "visibility_km",
            "precip_mm",
            "uv_index"
        ]

        with left:
            x_axis = st.selectbox(
                "X-Axis",
                numerical_features,
                key="x_axis"
            )

        with right:
            y_axis = st.selectbox(
                "Y-Axis",
                numerical_features,
                index=1,
                key="y_axis"
            )

        fig = px.scatter(
            filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            ),
            x=x_axis,
            y=y_axis,
            color="temperature_celsius",
            template="plotly_white",
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    with corr_tab:

        corr_features = [
            "temperature_celsius",
            "humidity",
            "pressure_mb",
            "wind_kph",
            "visibility_km",
            "precip_mm",
            "uv_index",
            "cloud",
            "air_quality_PM2.5",
            "air_quality_PM10"
        ]

        corr = filtered_df[corr_features].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        st.pyplot(fig)

    with time_tab:
        left, right = st.columns(2)
        with left:
            monthly = (
                filtered_df
                .groupby("month")["temperature_celsius"]
                .mean()
                .reset_index()
            )
            fig = px.line(
                monthly,
                x="month",
                y="temperature_celsius",
                markers=True,
                template="plotly_white",
                title="Monthly Temperature Trend"
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            seasonal = (
                filtered_df
                .groupby("season")["temperature_celsius"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                seasonal,
                x="season",
                y="temperature_celsius",
                color="season",
                template="plotly_white",
                title="Seasonal Temperature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with insight_tab:
        st.subheader("Key Observations")
        insights = [
            "Tree-based models perform significantly better than linear regression, indicating nonlinear relationships among weather variables.",
            "Temperature exhibits clear seasonal and monthly variation.",
            "Humidity and temperature show a noticeable inverse relationship in many regions.",
            "Air quality metrics display meaningful correlations with weather conditions.",
            "The dataset covers a wide geographic range, enabling robust global weather analysis."
        ]

        for item in insights:
            st.markdown(f"- {item}")
        st.divider()
        st.subheader("Top 10 Warmest Countries")
        warmest = (
            filtered_df
            .groupby("country")["temperature_celsius"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig = px.bar(
            warmest,
            x="temperature_celsius",
            y="country",
            orientation="h",
            template="plotly_white",
            title="Average Temperature by Country"
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

def machine_learning():
    st.title("Machine Learning")
    st.caption(
        "Comparison of forecasting models used to predict weather trends."
    )
    overview_tab, comparison_tab, importance_tab, conclusion_tab = st.tabs(
        [
            "Overview",
            "Model Comparison",
            "Feature Importance",
            "Conclusion"
        ]
    )
    with overview_tab:
        best_model = results_df.loc[
            results_df["R2"].idxmax()
        ]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card(
                "Models Trained",
                len(results_df)
            )
        with c2:
            kpi_card(
                "Best Model",
                best_model["Model"]
            )
        with c3:
            kpi_card(
                "Best R²",
                round(best_model["R2"], 3)
            )
        with c4:
            kpi_card(
                "Best RMSE",
                round(best_model["RMSE"], 3)
            )
        st.divider()
        st.subheader("Model Evaluation Results")
        st.dataframe(
            results_df.style.format({
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "R2": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )


    with comparison_tab:
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                results_df,
                x="Model",
                y="R2",
                color="Model",
                text="R2",
                title="R² Comparison",
                template="plotly_white"
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.bar(
                results_df,
                x="Model",
                y="RMSE",
                color="Model",
                text="RMSE",
                title="RMSE Comparison",
                template="plotly_white"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        fig = px.bar(
            results_df,
            x="Model",
            y="MAE",
            color="Model",
            text="MAE",
            title="MAE Comparison",
            template="plotly_white"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with importance_tab:

        st.subheader("Top 20 Most Important Features")

        top_features = (
            importance_df.sort_values(
                "Importance",
                ascending=False
            ).head(20)
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            template="plotly_white",
            title="XGBoost Feature Importance"
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True
        )

    with conclusion_tab:

        st.subheader("Model Summary")

        st.success(
            f"""
            Best Performing Model: **{best_model['Model']}**

            R² Score: **{best_model['R2']:.3f}**

            RMSE: **{best_model['RMSE']:.3f}**

            MAE: **{best_model['MAE']:.3f}**
            """
        )

        st.divider()

        st.subheader("Key Findings")

        st.markdown("""
        - XGBoost achieved the highest predictive performance among all evaluated models.
        - The Ensemble model produced performance close to XGBoost but did not surpass it.
        - Random Forest significantly outperformed Linear Regression, indicating nonlinear relationships within the weather data.
        - Feature importance analysis shows that UV Index, Season, Pressure, Latitude, and Humidity are among the strongest predictors of temperature.
        - Ensemble learning improved robustness while maintaining high forecasting accuracy.
        """)

        st.divider()

        st.subheader("Model Ranking")

        ranking = (
            results_df
            .sort_values(
                "R2",
                ascending=False
            )
            .reset_index(drop=True)
        )

        ranking.index += 1

        st.dataframe(
            ranking,
            use_container_width=True
        )

def climate_analysis():

    st.title("Climate Analysis")
    st.caption(
        "Analysis of long-term weather patterns across months, years, seasons, and countries."
    )

    overview_tab, monthly_tab, yearly_tab, seasonal_tab, country_tab = st.tabs(
        [
            "Overview",
            "Monthly Trends",
            "Yearly Trends",
            "Seasonal Analysis",
            "Country Comparison"
        ]
    )

    with overview_tab:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Avg Temperature",
                f"{filtered_df['temperature_celsius'].mean():.2f} °C"
            )

        with c2:
            kpi_card(
                "Highest Temperature",
                f"{filtered_df['temperature_celsius'].max():.2f} °C"
            )

        with c3:
            kpi_card(
                "Lowest Temperature",
                f"{filtered_df['temperature_celsius'].min():.2f} °C"
            )

        with c4:
            kpi_card(
                "Countries",
                filtered_df["country"].nunique()
            )

        st.divider()

        overview = (
            filtered_df
            .groupby("season")["temperature_celsius"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        overview.columns = [
            "Season",
            "Average",
            "Minimum",
            "Maximum"
        ]

        st.dataframe(
            overview.style.format({
                "Average": "{:.2f}",
                "Minimum": "{:.2f}",
                "Maximum": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with monthly_tab:

        monthly = (
            filtered_df
            .groupby("month")["temperature_celsius"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="month",
            y="temperature_celsius",
            markers=True,
            title="Average Monthly Temperature",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        monthly_precip = (
            filtered_df
            .groupby("month")["precip_mm"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            monthly_precip,
            x="month",
            y="precip_mm",
            title="Average Monthly Precipitation",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with yearly_tab:

        yearly = (
            filtered_df
            .groupby("year")[
                [
                    "temperature_celsius",
                    "humidity"
                ]
            ]
            .mean()
            .reset_index()
        )

        fig = px.line(
            yearly,
            x="year",
            y="temperature_celsius",
            markers=True,
            title="Average Yearly Temperature",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.line(
            yearly,
            x="year",
            y="humidity",
            markers=True,
            title="Average Yearly Humidity",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with seasonal_tab:

        seasonal = (
            filtered_df
            .groupby("season")[
                [
                    "temperature_celsius",
                    "humidity",
                    "precip_mm"
                ]
            ]
            .mean()
            .reset_index()
        )

        left, right = st.columns(2)

        with left:

            fig = px.bar(
                seasonal,
                x="season",
                y="temperature_celsius",
                color="season",
                title="Seasonal Temperature",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.bar(
                seasonal,
                x="season",
                y="humidity",
                color="season",
                title="Seasonal Humidity",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        fig = px.bar(
            seasonal,
            x="season",
            y="precip_mm",
            color="season",
            title="Seasonal Precipitation",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with country_tab:

        top_n = st.slider(
            "Number of Countries",
            5,
            25,
            10
        )

        ranking = (
            filtered_df
            .groupby("country")["temperature_celsius"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            ranking,
            x="temperature_celsius",
            y="country",
            orientation="h",
            color="temperature_celsius",
            title="Warmest Countries",
            template="plotly_white"
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        coldest = (
            filtered_df
            .groupby("country")["temperature_celsius"]
            .mean()
            .sort_values()
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            coldest,
            x="temperature_celsius",
            y="country",
            orientation="h",
            color="temperature_celsius",
            title="Coldest Countries",
            template="plotly_white"
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

def air_quality():

    st.title("Air Quality Analysis")
    st.caption(
        "Analysis of air quality indicators and their relationship with weather conditions."
    )

    overview_tab, pollutant_tab, correlation_tab, country_tab, index_tab = st.tabs(
        [
            "Overview",
            "Pollutants",
            "Correlations",
            "Country Analysis",
            "Air Quality Index"
        ]
    )

    with overview_tab:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Avg PM2.5",
                f"{filtered_df['air_quality_PM2.5'].mean():.2f}"
            )

        with c2:
            kpi_card(
                "Avg PM10",
                f"{filtered_df['air_quality_PM10'].mean():.2f}"
            )

        with c3:
            kpi_card(
                "Avg EPA Index",
                f"{filtered_df['air_quality_us-epa-index'].mean():.2f}"
            )

        with c4:
            kpi_card(
                "Avg DEFRA Index",
                f"{filtered_df['air_quality_gb-defra-index'].mean():.2f}"
            )

        st.divider()

        summary = filtered_df[
            [
                "air_quality_PM2.5",
                "air_quality_PM10",
                "air_quality_Carbon_Monoxide",
                "air_quality_Ozone",
                "air_quality_Nitrogen_dioxide",
                "air_quality_Sulphur_dioxide"
            ]
        ].describe()

        st.dataframe(
            summary,
            use_container_width=True
        )
    with pollutant_tab:

        pollutant = st.selectbox(
            "Select Pollutant",
            [
                "air_quality_PM2.5",
                "air_quality_PM10",
                "air_quality_Carbon_Monoxide",
                "air_quality_Ozone",
                "air_quality_Nitrogen_dioxide",
                "air_quality_Sulphur_dioxide"
            ]
        )

        left, right = st.columns(2)

        with left:

            fig = px.histogram(
                filtered_df,
                x=pollutant,
                nbins=40,
                marginal="box",
                template="plotly_white",
                title=f"{pollutant.replace('_', ' ')} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.box(
                filtered_df,
                y=pollutant,
                template="plotly_white",
                title=f"{pollutant.replace('_', ' ')} Box Plot"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with correlation_tab:
        weather_feature = st.selectbox(
            "Weather Feature",
            [
                "temperature_celsius",
                "humidity",
                "pressure_mb",
                "wind_kph",
                "visibility_km",
                "cloud",
                "uv_index"
            ]
        )

        pollutant = st.selectbox(
            "Air Quality Feature",
            [
                "air_quality_PM2.5",
                "air_quality_PM10",
                "air_quality_Carbon_Monoxide",
                "air_quality_Ozone",
                "air_quality_Nitrogen_dioxide",
                "air_quality_Sulphur_dioxide"
            ],
            key="corr_pollutant"
        )

        fig = px.scatter(
            filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            ),
            x=weather_feature,
            y=pollutant,
            color="temperature_celsius",
            template="plotly_white",
            title=f"{weather_feature} vs {pollutant}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        corr = filtered_df[
            [
                weather_feature,
                pollutant
            ]
        ].corr().iloc[0, 1]

        st.info(
            f"Correlation Coefficient: {corr:.3f}"
        )
    with country_tab:

        metric = st.selectbox(
            "Ranking Metric",
            [
                "air_quality_PM2.5",
                "air_quality_PM10",
                "air_quality_Carbon_Monoxide",
                "air_quality_Ozone"
            ]
        )

        top_n = st.slider(
            "Top Countries",
            5,
            20,
            10
        )

        ranking = (
            filtered_df
            .groupby("country")[metric]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        fig = px.bar(
            ranking,
            x=metric,
            y="country",
            orientation="h",
            color=metric,
            template="plotly_white",
            title=f"Highest {metric.replace('_', ' ')}"
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    with index_tab:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                filtered_df,
                x="air_quality_us-epa-index",
                template="plotly_white",
                title="US EPA Air Quality Index"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            fig = px.histogram(
                filtered_df,
                x="air_quality_gb-defra-index",
                template="plotly_white",
                title="GB DEFRA Air Quality Index"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        index_summary = (
            filtered_df[
                [
                    "air_quality_us-epa-index",
                    "air_quality_gb-defra-index"
                ]
            ]
            .describe()
        )
        st.dataframe(
            index_summary,
            use_container_width=True
        )


def spatial_analysis():

    st.title("Spatial Analysis")
    st.caption(
        "Interactive visualization of global weather and air quality patterns."
    )

    overview_tab, temperature_tab, air_tab, weather_tab, insights_tab = st.tabs(
        [
            "Overview",
            "Temperature Map",
            "Air Quality Map",
            "Weather Map",
            "Geographic Insights"
        ]
    )
    with overview_tab:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Countries",
                filtered_df["country"].nunique()
            )

        with c2:
            kpi_card(
                "Cities",
                filtered_df["location_name"].nunique()
            )

        with c3:
            kpi_card(
                "Latitude Range",
                f"{filtered_df['latitude'].min():.1f} → {filtered_df['latitude'].max():.1f}"
            )

        with c4:
            kpi_card(
                "Longitude Range",
                f"{filtered_df['longitude'].min():.1f} → {filtered_df['longitude'].max():.1f}"
            )

        st.divider()

        st.dataframe(
            filtered_df[
                [
                    "country",
                    "location_name",
                    "latitude",
                    "longitude",
                    "temperature_celsius",
                    "humidity",
                    "air_quality_PM2.5"
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )

    with temperature_tab:
        fig = px.scatter_geo(
            filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            ),
            lat="latitude",
            lon="longitude",
            color="temperature_celsius",
            size="humidity",
            hover_name="location_name",
            hover_data=[
                "country",
                "temperature_celsius",
                "humidity",
                "wind_kph"
            ],
            projection="natural earth",
            template="plotly_white",
            title="Global Temperature Distribution"
        )

        fig.update_layout(height=700)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with air_tab:

        pollutant = st.selectbox(
            "Air Quality Metric",
            [
                "air_quality_PM2.5",
                "air_quality_PM10",
                "air_quality_Carbon_Monoxide",
                "air_quality_Ozone",
                "air_quality_Nitrogen_dioxide",
                "air_quality_Sulphur_dioxide"
            ]
        )

        fig = px.scatter_geo(
            filtered_df.sample(
                min(5000, len(filtered_df)),
                random_state=42
            ),
            lat="latitude",
            lon="longitude",
            color=pollutant,
            size="humidity",
            hover_name="location_name",
            hover_data=[
                "country",
                pollutant
            ],
            projection="natural earth",
            template="plotly_white",
            title=f"{pollutant.replace('_', ' ')} Distribution"
        )

        fig.update_layout(height=700)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with weather_tab:

        metric = st.selectbox(
            "Weather Metric",
            [
                "temperature_celsius",
                "humidity",
                "wind_kph",
                "pressure_mb",
                "cloud",
                "visibility_km",
                "uv_index"
            ]
        )

        map_df = filtered_df.sample(
            min(5000, len(filtered_df)),
            random_state=42
        )

        fig = px.scatter_geo(
            map_df,
            lat="latitude",
            lon="longitude",
            color=metric,
            size="humidity",
            hover_name="location_name",
            hover_data=["country", metric],
            projection="natural earth",
            template="plotly_white",
            title=f"Global {metric.replace('_', ' ').title()}"
        )

        fig.update_layout(height=700)

        st.plotly_chart(fig, use_container_width=True)

    with insights_tab:

        left, right = st.columns(2)

        with left:

            hottest = (
                filtered_df
                .groupby("country")["temperature_celsius"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            fig = px.bar(
                hottest,
                x="temperature_celsius",
                y="country",
                orientation="h",
                color="temperature_celsius",
                template="plotly_white",
                title="Warmest Countries"
            )

            fig.update_layout(
                yaxis=dict(categoryorder="total ascending")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            polluted = (
                filtered_df
                .groupby("country")["air_quality_PM2.5"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            fig = px.bar(
                polluted,
                x="air_quality_PM2.5",
                y="country",
                orientation="h",
                color="air_quality_PM2.5",
                template="plotly_white",
                title="Highest Average PM2.5"
            )

            fig.update_layout(
                yaxis=dict(categoryorder="total ascending")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        st.subheader("Geographical Insights")

        st.markdown("""
- The dataset spans more than **200 countries**, enabling global-scale weather analysis.
- Temperature varies considerably across latitudes, highlighting climatic differences between regions.
- Air quality metrics show notable regional variation, with some countries exhibiting consistently higher PM2.5 values.
- Weather variables such as humidity and wind speed display clear spatial patterns when viewed geographically.
- Interactive maps provide a quick way to identify regional climate characteristics and environmental conditions.
""")


def anomaly_detection():
    iso_model = joblib.load(ISO_MODEL)
    st.title("Anomaly Detection")
    st.caption(
        "Detection and analysis of unusual weather observations using Isolation Forest."
    )

    overview_tab, distribution_tab, analysis_tab, records_tab, insights_tab = st.tabs(
        [
            "Overview",
            "Distribution",
            "Analysis",
            "Records",
            "Insights"
        ]
    )

    anomaly_df = filtered_df.copy()

    features = [
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "precip_mm"
    ]

    anomaly_df["anomaly"] = iso_model.predict(
        anomaly_df[features]
    )
    normal_count = (anomaly_df["anomaly"] == 1).sum()
    anomaly_count = (anomaly_df["anomaly"] == -1).sum()

    with overview_tab:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card(
                "Total Records",
                f"{len(anomaly_df):,}"
            )

        with c2:
            kpi_card(
                "Normal",
                f"{normal_count:,}"
            )

        with c3:
            kpi_card(
                "Anomalies",
                f"{anomaly_count:,}"
            )

        with c4:

            anomaly_rate = anomaly_count / len(anomaly_df) * 100

            kpi_card(
                "Anomaly Rate",
                f"{anomaly_rate:.2f}%"
            )

        st.divider()

        summary = pd.DataFrame(
            {
                "Category": [
                    "Normal",
                    "Anomaly"
                ],
                "Count": [
                    normal_count,
                    anomaly_count
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    with distribution_tab:

        fig = px.pie(
            summary,
            names="Category",
            values="Count",
            hole=0.55,
            template="plotly_white",
            title="Normal vs Anomalous Records"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.histogram(
            anomaly_df,
            x="temperature_celsius",
            color="anomaly",
            nbins=40,
            template="plotly_white",
            title="Temperature Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with analysis_tab:

        feature = st.selectbox(
            "Select Feature",
            [
                "temperature_celsius",
                "humidity",
                "pressure_mb",
                "wind_kph",
                "visibility_km",
                "precip_mm",
                "cloud",
                "uv_index"
            ]
        )

        fig = px.box(
            anomaly_df,
            x="anomaly",
            y=feature,
            color="anomaly",
            template="plotly_white",
            title=f"{feature.replace('_', ' ').title()} by Anomaly Class"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.scatter(
            anomaly_df.sample(
                min(5000, len(anomaly_df)),
                random_state=42
            ),
            x="humidity",
            y="temperature_celsius",
            color="anomaly",
            template="plotly_white",
            title="Humidity vs Temperature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with records_tab:

        anomalies = anomaly_df[
            anomaly_df["anomaly"] == -1
        ]

        display_cols = [
            "country",
            "location_name",
            "last_updated",
            "temperature_celsius",
            "humidity",
            "pressure_mb",
            "wind_kph",
            "condition_text"
        ]

        st.subheader("Detected Anomalies")

        st.dataframe(
            anomalies[display_cols],
            use_container_width=True,
            hide_index=True
        )

        csv = anomalies.to_csv(index=False)

        st.download_button(
            "Download Anomaly Records",
            csv,
            file_name="anomaly_records.csv",
            mime="text/csv"
        )
    with insights_tab:

        st.subheader("Key Findings")

        st.markdown(f"""
- Isolation Forest detected **{anomaly_count:,}** anomalous observations.
- Approximately **{anomaly_rate:.2f}%** of all observations were classified as anomalies.
- Most records were classified as normal, indicating a generally consistent weather dataset.
- Outliers are typically associated with extreme temperature, humidity, pressure, or wind conditions.
- These anomalies may represent rare weather events, measurement errors, or exceptional climatic conditions.
        """)

        st.divider()

        st.subheader("Recommendation")

        st.info(
            """
            Anomaly detection is valuable for identifying rare weather events,
            improving forecasting reliability, and detecting abnormal environmental
            conditions before model training.
            """
        )

def about():
    st.title("About")

    st.markdown("""
    ### Weather Trend Forecasting

    This project was developed as part of the **PM Accelerator Technical Assessment**.

    **Author:** Immanuel G

    **Dataset:** Global Weather Repository (Kaggle)

    **Technologies Used**
    - Python
    - Pandas
    - Plotly
    - Scikit-learn
    - XGBoost
    - Streamlit

    **GitHub**
    https://github.com/Immanuel2004

    **LinkedIn**
    https://www.linkedin.com/in/immanuelg01/

    **Email**
    immanueljoshua35@gmail.com
    """)


if page == "Dashboard":
    dashboard()
elif page == "Data Cleaning":
    data_cleaning()
elif page == "EDA":
    eda()
elif page == "Machine Learning":
    machine_learning()
elif page == "Climate Analysis":
    climate_analysis()
elif page == "Air Quality":
    air_quality()
elif page == "Spatial Analysis":
    spatial_analysis()
elif page == "Anomaly Detection":
    anomaly_detection()
else:
    about()
