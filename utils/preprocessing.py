import streamlit as st

def preprocess_data(df):
    st.subheader("🧹 Preprocessing")

    method = st.selectbox("Missing Values", ["None", "Drop", "Fill Mean"])

    before = df.copy()

    if method == "Drop":
        df = df.dropna()
    elif method == "Fill Mean":
        df = df.fillna(df.mean(numeric_only=True))

    st.write("### Before")
    st.write(before.head())

    st.write("### After")
    st.write(df.head())

    return df
