import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    st.write("Summary Statistics")
    st.write(df.describe())

    st.write("### Class Distribution")
    target = df.columns[-1]
    st.bar_chart(df[target].value_counts())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
    st.pyplot(fig)
