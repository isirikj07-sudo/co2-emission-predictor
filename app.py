import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CO2 Emission Predictor", page_icon="🌿", layout="wide")

@st.cache_resource
def load_all():
    with open("model_lr.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("model_rf.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        fc = pickle.load(f)
    with open("dummy_columns.pkl", "rb") as f:
        dc = pickle.load(f)
    with open("original_categories.pkl", "rb") as f:
        oc = pickle.load(f)
    return lr, rf, scaler, fc, dc, oc

@st.cache_data
def load_data():
    df = pd.read_csv("vehicaLCO2EMISSION.csv")
    return df.drop_duplicates().dropna()

lr_model, rf_model, scaler, feature_cols, dummy_cols, original_categories = load_all()
df = load_data()

st.sidebar.title("🌿 CO2 Predictor")
st.sidebar.markdown("**SRN:** PES1PG25CA282")
st.sidebar.markdown("**Name:** ISIRI K J")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "📊 EDA", "ℹ️ About"])

if page == "🏠 Home":
    st.title("🌍 Vehicle CO2 Emission Predictor")
    st.markdown("**Climate Action using Machine Learning — SDG 13**")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg CO2 (g/km)", f"{df['CO2 Emissions(g/km)'].mean():.1f}")
    c3.metric("Min CO2", f"{df['CO2 Emissions(g/km)'].min()}")
    c4.metric("Max CO2", f"{df['CO2 Emissions(g/km)'].max()}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CO2 Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["CO2 Emissions(g/km)"], bins=40, color="steelblue", edgecolor="white")
        ax.axvline(150, color="green", linestyle="--", label="Low < 150")
        ax.axvline(250, color="red", linestyle="--", label="High > 250")
        ax.set_xlabel("CO2 (g/km)")
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("Avg CO2 by Fuel Type")
        fig2, ax2 = plt.subplots()
        fuel_avg = df.groupby("Fuel Type")["CO2 Emissions(g/km)"].mean().sort_values()
        ax2.barh(fuel_avg.index, fuel_avg.values, color="coral")
        ax2.set_xlabel("Avg CO2 (g/km)")
        st.pyplot(fig2)

elif page == "🔍 Predict":
    st.title("🔍 Predict CO2 Emission")
    st.markdown("All vehicle features — including Make, Class, Transmission, and Fuel Type — now meaningfully influence the prediction.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        make      = st.selectbox("Make",          original_categories["Make"])
        veh_class = st.selectbox("Vehicle Class", original_categories["Vehicle Class"])
        fuel_type = st.selectbox("Fuel Type",     original_categories["Fuel Type"])

    with col2:
        engine_size  = st.slider("Engine Size (L)", 0.9, 8.4, 2.0, 0.1)
        cylinders    = st.selectbox("Cylinders",    sorted(df["Cylinders"].unique()))
        transmission = st.selectbox("Transmission", original_categories["Transmission"])

    with col3:
        fuel_city = st.slider("Fuel City (L/100km)",  3.0, 30.0, 10.0, 0.1)
        fuel_hwy  = st.slider("Fuel Hwy (L/100km)",   3.0, 25.0,  8.0, 0.1)
        fuel_comb = st.slider("Fuel Comb (L/100km)",  3.0, 27.0,  9.0, 0.1)
        fuel_mpg  = st.slider("Fuel Comb (mpg)",      10,  70,    30)

    model_choice = st.radio("Choose Model", ["Linear Regression", "Random Forest"], horizontal=True)

    if st.button("🚀 Predict", type="primary"):
        # Start with all dummy columns set to 0
        row = {col: 0 for col in dummy_cols}

        # Set the relevant one-hot flags to 1
        def set_dummy(prefix, value):
            col_name = f"{prefix}_{value}"
            if col_name in row:
                row[col_name] = 1

        set_dummy("Make", make)
        set_dummy("Vehicle Class", veh_class)
        set_dummy("Transmission", transmission)
        set_dummy("Fuel Type", fuel_type)

        # Add numeric features
        row["Engine Size(L)"] = engine_size
        row["Cylinders"] = cylinders
        row["Fuel Consumption City (L/100 km)"] = fuel_city
        row["Fuel Consumption Hwy (L/100 km)"] = fuel_hwy
        row["Fuel Consumption Comb (L/100 km)"] = fuel_comb
        row["Fuel Consumption Comb (mpg)"] = fuel_mpg

        input_df     = pd.DataFrame([row])[feature_cols]
        input_scaled = scaler.transform(input_df)

        if model_choice == "Linear Regression":
            prediction = lr_model.predict(input_scaled)[0]
        else:
            prediction = rf_model.predict(input_scaled)[0]

        if prediction < 150:
            zone, color, msg = "🟢 LOW",    "green",  "✅ Eco-friendly vehicle!"
        elif prediction < 250:
            zone, color, msg = "🟡 MEDIUM", "orange", "⚠️ Moderate emissions."
        else:
            zone, color, msg = "🔴 HIGH",   "red",    "❌ High emitter!"

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Predicted CO2 (g/km)", f"{prediction:.1f}")
        r2.markdown(f"### Emission Zone: {zone}")

        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(["CO2"], [prediction], color=color, height=0.4)
        ax.set_xlim(0, 600)
        ax.axvline(150, color="green", linestyle="--", linewidth=1)
        ax.axvline(250, color="red",   linestyle="--", linewidth=1)
        ax.set_xlabel("CO2 (g/km)")
        st.pyplot(fig)

        if color == "green":
            st.success(msg)
        elif color == "orange":
            st.warning(msg)
        else:
            st.error(msg)

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Top Makes"])
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df["CO2 Emissions(g/km)"], bins=40, color="steelblue", edgecolor="white")
        axes[0].set_title("CO2 Distribution")
        axes[1].hist(df["Engine Size(L)"], bins=30, color="coral", edgecolor="white")
        axes[1].set_title("Engine Size Distribution")
        st.pyplot(fig)
    with tab2:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    with tab3:
        n = st.slider("How many makes?", 5, 20, 10)
        top = df.groupby("Make")["CO2 Emissions(g/km)"].mean().sort_values(ascending=False).head(n)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(top.index, top.values, color="tomato")
        ax.set_xticklabels(top.index, rotation=45, ha="right")
        ax.set_ylabel("Avg CO2 (g/km)")
        plt.tight_layout()
        st.pyplot(fig)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    | Field | Info |
    |-------|------|
    | **SRN** | PES1PG25CA282 |
    | **Name** | ISIRI K J |
    | **Domain** | Environment & Climate |
    | **Goal** | SDG 13 — Climate Action |
    | **Models** | Linear Regression, Random Forest |
    | **Encoding** | One-Hot Encoding (categorical), Standard Scaler (numeric) |
    | **UI** | Streamlit |
    | **Dataset** | 7,385 vehicle records |
    """)
    st.subheader("Emission Zone Thresholds")
    st.markdown("""
    - 🟢 Low — CO2 < 150 g/km
    - 🟡 Medium — 150 to 250 g/km
    - 🔴 High — CO2 > 250 g/km
    """)
