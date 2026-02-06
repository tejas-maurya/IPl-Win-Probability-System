import streamlit as st
import pickle
import pandas as pd

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

pipe = load_model()

# ------------------ Teams & Cities ------------------
teams = sorted([
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
])

cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# ------------------ Title ------------------
st.title("🏏 IPL Win Probability Predictor")
st.markdown("Predict **live match winning chances** using Machine Learning")
st.divider()

# ------------------ Team Selection ------------------
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)

with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", teams)

if batting_team == bowling_team:
    st.error("Batting and Bowling teams must be different!")
    st.stop()

# ------------------ City ------------------
city = st.selectbox("📍 Match City", cities)

# ------------------ Match Situation ------------------
st.subheader("📊 Match Situation")

col3, col4 = st.columns(2)

with col3:
    target = st.number_input("🎯 Target Score", min_value=1)

with col4:
    score = st.number_input("🏏 Current Score", min_value=0)

col5, col6, col7 = st.columns(3)

with col5:
    overs = st.number_input(
        "⏱ Overs Completed",
        min_value=0.1,
        max_value=20.0,
        step=0.1
    )

with col6:
    wickets_fallen = st.number_input(
        "❌ Wickets Fallen",
        min_value=0,
        max_value=10
    )

balls_completed = int(overs * 6)
balls_left = max(0, 120 - balls_completed)

with col7:
    st.metric("🏏 Balls Left", balls_left)

# ------------------ Prediction ------------------
st.divider()

if st.button("🔮 Predict Win Probability", use_container_width=True):

    if score >= target:
        st.success(f"🏆 {batting_team} has already won the match!")
        st.stop()

    runs_left = target - score
    wickets_remaining = 10 - wickets_fallen

    # Safe calculations
    curr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # 🔥 IMPORTANT: Column names match training data
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'ball_left': [balls_left],   # ✔ correct
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'curr': [curr],              # ✔ correct
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)[0]

    win_prob = round(result[1] * 100, 2)
    loss_prob = round(result[0] * 100, 2)

    st.subheader("📈 Winning Chances")

    col8, col9 = st.columns(2)

    with col8:
        st.success(f"🏆 {batting_team}")
        st.progress(win_prob / 100)
        st.metric("Win Probability", f"{win_prob}%")

    with col9:
        st.error(f"💥 {bowling_team}")
        st.progress(loss_prob / 100)
        st.metric("Win Probability", f"{loss_prob}%")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit & Machine Learning")
