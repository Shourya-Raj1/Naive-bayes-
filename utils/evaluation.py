import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    st.subheader("📈 Evaluation")

    st.metric("Accuracy", accuracy_score(y_test, y_pred))

    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
