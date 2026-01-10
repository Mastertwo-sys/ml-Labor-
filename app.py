import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📱 Mobil ML-Labor (scikit-learn)")

uploaded_file = st.file_uploader("CSV hochladen (Features + Label)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
else:
    st.info("Iris-Daten")
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)

epochs = st.slider("Runden", 1, 20, 10)
if st.button("Trainieren"):
    acc_history = []
    for epoch in range(epochs):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        acc_history.append(acc)
        st.write(f"Runde {epoch+1}: Acc = {acc:.3f}")
    st.line_chart(acc_history)