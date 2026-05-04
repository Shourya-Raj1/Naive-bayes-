import streamlit as st
import pandas as pd


from utils.eda import show_eda
from utils.model import train_model, predict_sample
from utils.evaluation import evaluate_model
from utils.learning import show_learning

st.set_page_config(layout="wide")

st.title("🧠 Naive Bayes Learning System")

tabs = st.tabs([
    "📂 Upload",
    "🧹 Preprocess",
    "📊 EDA",
    "📘 Learn",
    "⚙ Train",
    "🔮 Predict",
    "📈 Evaluate"
])

if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- UPLOAD ----------------
with tabs[0]:
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.write(df.head())
        st.write("Shape:", df.shape)

# ---------------- PREPROCESS ----------------
with tabs[1]:
    if st.session_state.df is not None:
        st.session_state.df = preprocess_data(st.session_state.df)

# ---------------- EDA ----------------
with tabs[2]:
    if st.session_state.df is not None:
        show_eda(st.session_state.df)

# ---------------- LEARNING ----------------
with tabs[3]:
    show_learning()

# ---------------- TRAIN ----------------
with tabs[4]:
    if st.session_state.df is not None:
        target = st.selectbox("Select Target", st.session_state.df.columns)
        model_type = st.selectbox("Model Type", ["Gaussian", "Multinomial"])

        if st.button("Train Model"):
            model, X_test, y_test, feature_names = train_model(
                st.session_state.df,
                target,
                model_type
            )

            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names

            st.success("Model Trained Successfully")

# ---------------- PREDICT ----------------
with tabs[5]:
    if "model" in st.session_state:
        st.subheader("Enter Input")

        inputs = {}

        for col in st.session_state.feature_names:
            inputs[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):
            pred = predict_sample(
                st.session_state.model,
                inputs,
                st.session_state.feature_names
            )
            st.success(f"Prediction: {pred}")

# ---------------- EVALUATE ----------------
with tabs[6]:
    if "model" in st.session_state:
        evaluate_model(
            st.session_state.model,
            st.session_state.X_test,
            st.session_state.y_test
        )
