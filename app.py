import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Restaurant Rating Prediction",
    page_icon="🍽️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem;
}

header {
    visibility: hidden;
}

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

h1 {
    text-align: center;
    color: white !important;
    font-size: 52px !important;
    font-weight: bold !important;
}

p, label {
    color: white !important;
    font-size: 17px !important;
    font-weight: bold !important;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 3.2em;
}

.stButton>button:hover {
    background-color: #ff1e1e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Dataset.csv")

# Clean data
df = df.dropna()
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# ---------------- FEATURES ----------------
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

# ---------------- LABEL ENCODING ----------------
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

# ---------------- MODEL TRAINING ----------------
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# ---------------- UI ----------------
st.title("🍽️ Restaurant Rating Prediction")

st.write("Predict restaurant ratings using ML based on restaurant details.")

votes = st.number_input("⭐ Enter Number of Votes", min_value=0)

price = st.slider("💰 Price Range", 1, 4)

cost = st.number_input("🍴 Average Cost for Two", min_value=0)

# ---------------- PREDICTION ----------------
if st.button("Predict Rating"):

    # safe default values for categorical features
    input_data = pd.DataFrame([[
        0,   # Country Code (encoded)
        0,   # City (encoded)
        0,   # Cuisines (encoded)
        cost,
        0,   # Currency
        0,   # Table booking
        0,   # Online delivery
        price,
        votes
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"⭐ Predicted Restaurant Rating: {prediction[0]:.2f}")