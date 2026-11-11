import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Indonesia CO₂ Emissions Predictor", page_icon="🌍", layout="wide")

st.title("Indonesia CO₂ Emissions per Capita Predictor")
st.subheader("SDG 13: Climate Action")
st.write(
    "This project uses historical CO₂ emissions per capita data for Indonesia "
    "and a Linear Regression model to estimate a future value."
)

@st.cache_data
def load_data():
    data = pd.read_csv("co2_pcap_cons.csv")
    indonesia = data[data["name"].astype(str).str.lower() == "indonesia"].copy()
    if indonesia.empty:
        raise ValueError("Indonesia was not found in the CSV file.")

    year_columns = [column for column in data.columns if str(column).isdigit()]
    year_columns = sorted(year_columns, key=lambda column: int(column))

    long_data = indonesia.melt(
        id_vars=["geo", "name"],
        value_vars=year_columns,
        var_name="Year",
        value_name="CO2_per_capita"
    )
    long_data["Year"] = pd.to_numeric(long_data["Year"], errors="coerce")
    long_data["CO2_per_capita"] = pd.to_numeric(
        long_data["CO2_per_capita"], errors="coerce"
    )
    long_data = (
        long_data.dropna(subset=["Year", "CO2_per_capita"])
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return long_data

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.info("Make sure co2_pcap_cons.csv is uploaded to the same GitHub repository as app.py.")
    st.stop()

if len(df) < 5:
    st.error("There are not enough valid Indonesia data points to train and test the model.")
    st.stop()

first_year = int(df["Year"].min())
latest_year = int(df["Year"].max())

st.markdown("### Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("First year with data", first_year)
c2.metric("Latest year with data", latest_year)
c3.metric("Number of observations", len(df))

st.caption("The dataset is provided in the CSV file. Missing years are excluded from the model.")

st.markdown("### 1. Historical CO₂ emissions per capita")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df["Year"], df["CO2_per_capita"], marker=".", linewidth=1.6)
ax.set_title("Indonesia: CO₂ Emissions per Capita Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("CO₂ per capita (dataset value)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("### 2. Train and evaluate the model")
X = df[["Year"]]
y = df["CO2_per_capita"]
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.4f}")
m2.metric("RMSE", f"{rmse:.4f}")
m3.metric("R²", f"{r2:.4f}")

st.caption(
    "MAE and RMSE measure prediction error (lower is better). "
    "R² describes how well the model explains variation in the test data; "
    "it can be negative."
)

comparison = pd.DataFrame({
    "Year": X_test["Year"].values,
    "Actual": y_test.values,
    "Predicted": test_predictions
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

fig2, ax2 = plt.subplots(figsize=(11, 4))
ax2.plot(X_train["Year"], y_train, label="Training data")
ax2.plot(X_test["Year"], y_test, marker="o", label="Actual test data")
ax2.plot(X_test["Year"], test_predictions, marker="o", linestyle="--", label="Predicted test data")
ax2.set_title("Actual vs Predicted CO₂ per Capita")
ax2.set_xlabel("Year")
ax2.set_ylabel("CO₂ per capita (dataset value)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("### 3. Predict a future year")
final_model = LinearRegression()
final_model.fit(X, y)

prediction_year = st.number_input(
    "Choose a year to estimate",
    min_value=latest_year + 1,
    max_value=latest_year + 30,
    value=latest_year + 1,
    step=1
)
future_value = final_model.predict(pd.DataFrame({"Year": [int(prediction_year)]}))[0]
st.metric(f"Estimated CO₂ per capita in {int(prediction_year)}", f"{future_value:.4f}")

st.warning(
    "This is an estimate based only on a straight-line trend in historical data. "
    "It is not a certain outcome and does not account for changes in energy use, "
    "population, industry, technology, or climate policy."
)

st.markdown("### 4. Connection to SDG 13")
st.write(
    "Monitoring CO₂ emissions can help people understand patterns related to climate change. "
    "This project demonstrates how historical data and machine learning can support discussion "
    "about climate action. The model is a simplified learning project, not an official forecast."
)

with st.expander("View the Indonesia data used by the model"):
    st.dataframe(df, use_container_width=True, hide_index=True)
