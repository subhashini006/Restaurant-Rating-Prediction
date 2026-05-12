import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Page Configuration
st.set_page_config(
    page_title="Restaurant Rating Prediction",
    page_icon="🍽️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

/* Remove top white space */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem;
}

/* Hide Streamlit Header */
header {
    visibility: hidden;
}

/* Background Image */
.stApp {
    background-image: linear-gradient(
        rgba(0, 20, 60, 0.88),
        rgba(0, 20, 60, 0.88)
    ),
    url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Main Container */
.main-box {
    padding: 30px;
    border-radius: 18px;
    margin-top: 20px;
}

/* Title */
h1 {
    text-align: center;
    color: white !important;
    font-size: 52px !important;
    font-weight: bold !important;
}

/* Description */
p {
    color: white !important;
    font-size: 18px !important;
}

/* Labels */
label {
    color: white !important;
    font-size: 17px !important;
    font-weight: bold !important;
}

/* Input Boxes */
.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* Slider */
.stSlider {
    color: white !important;
}

/* Button */
.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 3.2em;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    color: white;
}

/* Remove top decoration/box */
[data-testid="stDecoration"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# Load Dataset
df = pd.read_csv("dataset.csv")

# Fill Missing Values
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Features and Target
X = df[[
    "Country Code",
    "City",
    "Cuisines",
    "Average Cost for two",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
    "Price range",
    "Votes"
]]

y = df["Aggregate rating"]

# Encode Categorical Columns
le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X.loc[:, col] = le.fit_transform(X[col])

# Train Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Main Box
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Title
st.title("🍽️ Restaurant Rating Prediction")

# Description
st.write(
    "This Machine Learning web application predicts restaurant ratings "
    "based on restaurant details such as votes, price range, and average cost."
)

# Inputs
votes = st.number_input("⭐ Enter Number of Votes", min_value=0)

price = st.slider("💰 Select Price Range", 1, 4)

cost = st.number_input("🍴 Average Cost for Two", min_value=0)

# Prediction
if st.button("Predict Rating"):

    sample = X.iloc[0:1].copy()

    sample["Votes"] = votes
    sample["Price range"] = price
    sample["Average Cost for two"] = cost

    prediction = model.predict(sample)

    st.success(f"⭐ Predicted Restaurant Rating: {prediction[0]:.2f}")

st.markdown("</div>", unsafe_allow_html=True)
