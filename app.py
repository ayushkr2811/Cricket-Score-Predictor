import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgb_score_predictor.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.title("🏏 T20 Cricket Score Predictor")

batting_team = st.selectbox(
    "Batting Team",
    ["India","Australia","England","Pakistan","South Africa",
     "New Zealand","West Indies","Sri Lanka","Bangladesh","Afghanistan"]
)

bowling_team = st.selectbox(
    "Bowling Team",
    ["India","Australia","England","Pakistan","South Africa",
     "New Zealand","West Indies","Sri Lanka","Bangladesh","Afghanistan"]
)

overs = st.number_input("Overs Completed", min_value=5, max_value=20, value=10)
score = st.number_input("Current Score", min_value=0, value=80)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3)
crr = st.number_input("Current Run Rate", value=8.0)
runs_last_5 = st.number_input("Runs in Last 5 Overs", value=40)

if st.button("Predict Final Score"):
    input_data = {
        "OversCompleted": overs,
        "Score": score,
        "WicketsOut": wickets,
        "CurrentRunRate": crr,
        "RunsLast5Overs": runs_last_5
    }

    for col in model_columns:
        if col.startswith("BattingTeam_"):
            input_data[col] = 1 if col == f"BattingTeam_{batting_team}" else 0
        elif col.startswith("BowlingTeam_"):
            input_data[col] = 1 if col == f"BowlingTeam_{bowling_team}" else 0

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=model_columns, fill_value=0)

    remaining_runs = model.predict(df)[0]
    final_score = int(score + remaining_runs)

    st.success(f"Predicted Final Score: {final_score}")
