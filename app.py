import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# List of teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Streamlit app title
st.title('CrickWin')

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, step=1)

# Input fields for score, overs, and wickets
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)

with col4:
    overs = st.number_input('Overs completed', min_value=0.0, step=0.1)

with col5:
    wickets = st.number_input('Wickets left', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    try:
        # Ensure overs is greater than or equal to 0 to avoid division by zero
        if overs < 0:
            st.error('Overs must be greater than or equal to 0.')
        else:
            # Calculate current run rate (CRR)
            crr = score / overs if overs > 0 else 0

            # Calculate required run rate (RRR)
            balls_left = 120 - (overs * 6)
            runs_left = target - score
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # Create input DataFrame for the model
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Make prediction
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            # Display results
            st.header(f'Probability of {batting_team} Winning: {round(win * 100, 2)}%')
            st.header(f'Probability of {bowling_team} Winning: {round(loss * 100, 2)}%')

    except Exception as e:
        st.error(f'An error occurred: {e}')
