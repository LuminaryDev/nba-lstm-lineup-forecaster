import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer  # For manual binning

st.set_page_config(page_title="NBA Hybrid Lineup Forecaster", page_icon="🏀", layout="wide")

@st.cache_resource
def load_bn_model():
    """Load the Bayesian Network model."""
    with open("production_bayesian_network.pkl", "rb") as f:
        model = pickle.load(f)
    feature_info = json.load(open("production_feature_info.json"))
    return model, feature_info

@st.cache_resource
def load_or_train_lstm():
    """Mock LSTM for player time-series forecasting (PTS, AST, REB)."""
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=3, hidden_size=50, output_size=3):  # 3 inputs/outputs: PTS, AST, REB
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    # Mock pre-trained weights (in prod, load torch.save('lstm_model.pth'))
    model = SimpleLSTM()
    # Simulate training with dummy sequence data (replace with your real training)
    dummy_data = torch.randn(100, 5, 3)  # 100 samples, 5 timesteps, 3 features
    dummy_targets = torch.randn(100, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for _ in range(10):  # Quick "training" loop
        out = model(dummy_data)
        loss = criterion(out, dummy_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def compute_features_from_projections(projections, season_avgs):
    """Compute DELTA and FORM_RATIO features from LSTM outputs."""
    deltas = projections / season_avgs  # e.g., DELTA_PTS = predicted / avg
    form_ratios = np.clip(deltas, 0.5, 2.0)  # Normalized 0.5-2.0
    # Mock other features (in prod, compute from full lineup data)
    projection_strength = np.mean(form_ratios) * 10  # Scaled 0-20
    return {
        'PROJECTION_STRENGTH': projection_strength,
        'DELTA_PTS': deltas[0],
        'DELTA_AST': deltas[1],
        'DELTA_REB': deltas[2],
        # Add FG_PCT, PLUS_MINUS, etc., from user input or defaults
        'FG_PCT': 0.46,  # Default avg
        'PLUS_MINUS': 2.5,
        'LINEUP_NET_RATING_TALENT': 105,  # Default
        'AVG_FORM_RATIO_AST': form_ratios[1]
    }

def discretize_features(features):
    """Manual binning to levels (0-3) based on your notebook logic."""
    bins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    # For simplicity, hardcode quantiles from your data (e.g., DELTA_PTS: 0.85-2.04)
    delta_bins = np.array([[-np.inf, 0.9, 1.0, 1.1, np.inf]])  # Mock quantiles
    bins.fit(delta_bins.T)
    disc_features = {}
    for feat, val in features.items():
        if 'DELTA' in feat or 'RATIO' in feat:
            disc_features[f'{feat}_LEVEL'] = int(np.digitize(val, delta_bins[0]))
        elif 'STRENGTH' in feat:
            disc_features[f'{feat}_LEVEL'] = min(3, int(val / 5))  # Scale to 0-3
        else:
            disc_features[f'{feat}_LEVEL'] = 2  # Default avg
    return disc_features

def predict_lineup(bn_model, evidence):
    """Run BN inference."""
    infer = VariableElimination(bn_model)
    result = infer.query(variables=['LINEUP_QUALITY_SCORE_LEVEL'], evidence=evidence)
    probs = np.array([result.values[state] for state in [0, 1, 2]])
    pred_class = np.argmax(probs)
    return pred_class, probs

def main():
    st.title("🏀 NBA Hybrid Lineup Forecaster")
    st.markdown("**LSTM Time-Series + Bayesian Network • Predict Lineup Efficiency (70.57% Acc)**")

    bn_model, feature_info = load_bn_model()
    lstm_model = load_or_train_lstm()

    tab1, tab2 = st.tabs(["🔍 Quick Predict (Sliders)", "⚡ Advanced (LSTM + Players)"])

    with tab1:
        st.header("Quick Mode: Direct Feature Inputs")
        col1, col2 = st.columns(2)
        with col1:
            proj = st.slider("🔮 Projection Strength (0-3)", 0, 3, 2)
            fg = st.slider("🎯 FG% Level (0-3)", 0, 3, 2)
            pm = st.slider("📈 Plus/Minus (0-3)", 0, 3, 2)
        with col2:
            net = st.slider("⭐ Net Rating (0-3)", 0, 3, 2)
            ast = st.slider("🤝 Assist Form (0-3)", 0, 3, 2)

        if st.button("🧠 Predict Efficiency", type="primary"):
            evidence = {
                'PROJECTION_STRENGTH_LEVEL': proj,
                'FG_PCT_LEVEL': fg,
                'PLUS_MINUS_LEVEL': pm,
                'LINEUP_NET_RATING_TALENT_LEVEL': net,
                'AVG_FORM_RATIO_AST_LEVEL': ast
            }
            pred_class, probs = predict_lineup(bn_model, evidence)
            st.success(f"**Prediction:** {['Low', 'Medium', 'High'][pred_class]} Efficiency")
            st.write("**Probabilities:**")
            cols = st.columns(3)
            for i, prob in enumerate(probs):
                with cols[i]:
                    st.progress(prob)
                    st.caption(f"{['Low', 'Medium', 'High'][i]}: {prob*100:.1f}%")

    with tab2:
        st.header("Advanced Mode: LSTM Time-Series Forecasting")
        st.markdown("Upload recent stats for 5 players (CSV: columns=PTS,AST,REB; rows=last 5 games each).")
        
        uploaded_file = st.file_uploader("📁 Upload Player Stats CSV", type="csv")
        season_avgs = st.text_input("Season Averages (comma-sep: PTS,AST,REB per player avg)", "10,3,4")  # Default
        
        if uploaded_file is not None and st.button("🚀 Run LSTM Forecast & Predict", type="primary"):
            with st.spinner("Running LSTM..."):
                data = pd.read_csv(uploaded_file)
                # Assume data shape: 25 rows (5 players x 5 games), or reshape
                sequences = data.values.reshape(5, 5, 3).astype(np.float32)  # Player, time, features
                sequences_tensor = torch.tensor(sequences)
                
                projections = []
                for seq in sequences_tensor:
                    with torch.no_grad():
                        pred = lstm_model(seq.unsqueeze(0))  # Predict next "game"
                        projections.append(pred.squeeze().numpy())
                projections = np.mean(projections, axis=0)  # Avg lineup projection
                
                season_avgs = np.array([float(x) for x in season_avgs.split(',')])
                features = compute_features_from_projections(projections, season_avgs)
                evidence = discretize_features(features)
                
                # Filter to BN features
                bn_evidence = {k: v for k, v in evidence.items() if k in feature_info['features']}
                
                pred_class, probs = predict_lineup(bn_model, bn_evidence)
                
                st.success(f"**LSTM Projections:** PTS={projections[0]:.1f}, AST={projections[1]:.1f}, REB={projections[2]:.1f}")
                st.json(features)  # Show intermediate features
                st.success(f"**Final Prediction:** {['Low', 'Medium', 'High'][pred_class]} Efficiency")
                st.write("**Probabilities:**")
                cols = st.columns(3)
                for i, prob in enumerate(probs):
                    with cols[i]:
                        st.progress(prob)
                        st.caption(f"{['Low', 'Medium', 'High'][i]}: {prob*100:.1f}%")
                
                # Export
                results = {'features': features, 'evidence': evidence, 'probs': probs.tolist()}
                st.download_button("💾 Download Results (JSON)", json.dumps(results), "prediction.json")

if __name__ == "__main__":
    main()
