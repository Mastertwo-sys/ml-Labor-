import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import copy
import plotly.express as px
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

st.title("📱 Mobil ML-Labor (optimiert)")

st.markdown("""
- Ensemble (Main + Anti-Modell)  
- Confidence Filter (adaptiv)  
- Edge-Case Sammlung  
- Drift Detection (KS-Test)  
- Rollback mit Backup-Update  
- Echtzeit-Dashboard mit Progress
- Einfach Start/Stop via Buttons
""")

# Daten
uploaded_file = st.file_uploader("CSV hochladen (Features + Label)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(np.float32)
else:
    st.info("Real Iris-Daten (binary: setosa vs. others)")
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = (iris.target == 0).astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initiale Features für Drift
initial_X = np.copy(X)

# Modell
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sig(self.fc3(x))

ensemble = [SimpleNN(X.shape[1]) for _ in range(2)]
optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in ensemble]
backups = [copy.deepcopy(m.state_dict()) for m in ensemble]
criterion = nn.BCELoss()

# Funktionen
def train_epoch(dataloader, ensemble, optimizers, threshold=0.6):
    edge_cases = []
    for X_batch, y_batch in dataloader:
        preds = []
        for m in ensemble:
            m.eval()
            with torch.no_grad():
                preds.append(m(X_batch).squeeze())
        mean_pred = torch.stack(preds).mean(0)
        conf = torch.abs(mean_pred - 0.5) * 2
        confident_idx = conf >= threshold
        if confident_idx.sum() == 0:
            edge_cases.append((X_batch.numpy(), y_batch.numpy()))
            continue
        X_train = X_batch[confident_idx]
        y_train = y_batch[confident_idx]
        for i, (m, opt) in enumerate(zip(ensemble, optimizers)):
            m.train()
            opt.zero_grad()
            target = 1 - y_train if i == 1 else y_train
            loss = criterion(m(X_train).squeeze(), target)
            loss.backward()
            opt.step()
    return edge_cases

def detect_drift(current_X, initial_X):
    stat, p = ks_2samp(current_X.flatten(), initial_X.flatten())
    return p < 0.05

def evaluate(ensemble, dataloader):
    all_X, correct, total = [], 0, 0
    for X_batch, y_batch in dataloader:
        all_X.append(X_batch.numpy())
        preds = []
        for m in ensemble:
            m.eval()
            with torch.no_grad():
                preds.append(m(X_batch).squeeze())
        mean_pred = torch.stack(preds).mean(0)
        labels = (mean_pred >= 0.5).float()
        correct += (labels == y_batch).sum().item()
        total += y_batch.size(0)
    current_X = np.concatenate(all_X)
    return correct / total, current_X

# Training UI
epochs = st.slider("Epochen", 1, 20, 10)
threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.6)

if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0

progress_bar = st.progress(0)
status_text = st.empty()

if st.button("Training starten"):
    st.session_state.running = True
    st.session_state.current_epoch = 0

if st.button("Training stoppen"):
    st.session_state.running = False

all_edge_cases = []
if st.session_state.running and st.session_state.current_epoch < epochs:
    edge_cases = train_epoch(dataloader, ensemble, optimizers, threshold)
    acc, current_X = evaluate(ensemble, dataloader)
    drift = detect_drift(current_X, initial_X)
    status_text.write(f"Epoche {st.session_state.current_epoch + 1}: Acc={acc:.3f}, Drift={'Ja' if drift else 'Nein'}, Edges={len(edge_cases)}")
    all_edge_cases.extend(edge_cases)
    if acc >= 0.8:
        backups = [copy.deepcopy(m.state_dict()) for m in ensemble]
    elif acc < 0.7:
        for i, m in enumerate(ensemble):
            m.load_state_dict(backups[i])
        st.warning("Rollback ausgelöst")
    st.session_state.current_epoch += 1
    progress_bar.progress(st.session_state.current_epoch / epochs)
    st.rerun()  # Rerun für nächste Epoche

if st.session_state.current_epoch == epochs:
    st.session_state.running = False
    st.success("Training abgeschlossen")

if all_edge_cases:
    st.subheader("Edge-Case Heatmap")
    if all_edge_cases[0][0].size > 0:
        X_ec = all_edge_cases[0][0]
        fig = px.imshow(X_ec, color_continuous_scale="Viridis", labels={"x":"Features","y":"Samples"})
        st.plotly_chart(fig)

if st.button("Modelle backuppen"):
    for i, m in enumerate(ensemble):
        torch.save(m.state_dict(), f"model_{i}.pt")
    st.success("Gespeichert")
