import streamlit as st
import numpy as np

def show_learning():
    st.header("📘 Naive Bayes Learning Module")

    st.markdown("## Bayes Theorem")
    st.latex(r"P(C|X) = \frac{P(X|C) * P(C)}{P(X)}")

    with st.expander("📌 What is this?"):
        st.write("""
        - P(C|X): Posterior
        - P(X|C): Likelihood
        - P(C): Prior
        - P(X): Evidence
        """)

    st.markdown("## Step-by-Step Example")

    prior = 0.6
    likelihood = 0.7
    evidence = 0.5

    posterior = (likelihood * prior) / evidence

    st.write(f"Prior = {prior}")
    st.write(f"Likelihood = {likelihood}")
    st.write(f"Evidence = {evidence}")

    st.success(f"Posterior = {posterior:.2f}")
