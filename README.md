# Indonesia CO2 Emissions per Capita Predictor (SDG 13)

A Streamlit web app that uses the supplied `co2_pcap_cons.csv` dataset and Linear Regression to explore and estimate Indonesia's CO2 emissions per capita.

## Files
- `app.py` — Streamlit application
- `co2_pcap_cons.csv` — original dataset used by the app
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy using GitHub and Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `co2_pcap_cons.csv` to the repository's root folder.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will install the packages listed in `requirements.txt`.

The CSV must be in the same repository folder as `app.py`.

## Method
The app selects Indonesia from the CSV, reshapes the year columns into rows, trains a Linear Regression model using the earliest 80% of valid observations, evaluates it on the latest 20%, and fits a final model on all observations for the selected future-year estimate.

## Limitation
The model uses year as its only feature and assumes a linear trend. Its future estimates are not certain forecasts and do not account for policy, energy, population, or economic changes.
