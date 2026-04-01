import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# === Page setup ===
st.set_page_config(page_title="AquaLemna AI", layout="centered")
st.title("🌿 AquaLemna AI")
st.subheader("Duckweed-Based Water Remediation Optimizer")
st.markdown(
    "Adjust environmental parameters and optimize duckweed-based water remediation."
)

# === Inputs ===
temp = st.slider("Temperature (°C)", 0, 100, 25)  # changed from 10–35 to 0–100
ph = st.slider("pH", 0.0, 14.0, 7.0)            # changed from 5–9 to 0–14
pond = st.number_input("Pond Size (m²)", 100, 10000, 1000)
nutrients = st.slider("Nutrient Level (0–1)", 0.1, 1.0, 0.5)
duckweed = st.number_input("Initial Duckweed (g)", 10, 200, 50)

# === Train model on synthetic data ===
@st.cache_data(show_spinner=False)
def train_model():
    np.random.seed(42)
    n = 2000
    temperature = np.random.uniform(10, 35, n)
    pH = np.random.uniform(5, 9, n)
    pond_size = np.random.uniform(100, 10000, n)
    nutrients_arr = np.random.uniform(0.1, 1.0, n)
    duckweed_start = np.random.uniform(10, 200, n)

    def temp_factor(T):
        return np.exp(-((T - 25)**2) / 50)

    def pH_factor(pH):
        return np.exp(-((pH - 7)**2) / 1.5)

    def nutrient_saturation(N):
        Ks = 0.3
        return N / (Ks + N)

    growth_rate = temp_factor(temperature) * pH_factor(pH) * nutrient_saturation(nutrients_arr)

    optimal_duckweed = pond_size * nutrients_arr * 0.05 + np.random.normal(0, 5, n)
    remediation_time = pond_size / (growth_rate * duckweed_start * 10 + 1e-5) + np.random.normal(0, 2, n)

    df = pd.DataFrame({
        "temperature": temperature,
        "pH": pH,
        "pond_size": pond_size,
        "nutrients": nutrients_arr,
        "duckweed_start": duckweed_start,
        "optimal_duckweed": optimal_duckweed,
        "remediation_time": remediation_time
    })

    X = df[["temperature", "pH", "pond_size", "nutrients", "duckweed_start"]]
    y = df[["optimal_duckweed", "remediation_time"]]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
    model.fit(X, y)
    return model

model = train_model()

# === Make prediction ===
sample = pd.DataFrame([{
    "temperature": temp,
    "pH": ph,
    "pond_size": pond,
    "nutrients": nutrients,
    "duckweed_start": duckweed
}])

prediction = model.predict(sample)[0]

st.success("Optimization Complete!")
st.metric("Optimal Duckweed Needed (g)", f"{prediction[0]:.2f}")
st.metric("Remediation Time (days)", f"{prediction[1]:.2f}")

# === Optional growth curve visualization ===
days = np.arange(0, 21)
growth = duckweed * (1 + 0.05)**days  # simple growth curve
fig, ax = plt.subplots()
ax.plot(days, growth, label="Duckweed Growth")
ax.axhline(y=prediction[0], color='r', linestyle='--', label="Optimal Duckweed")
ax.set_xlabel("Days")
ax.set_ylabel("Duckweed Biomass (g)")
ax.legend()
st.pyplot(fig)
