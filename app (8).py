
import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination

st.set_page_config(page_title="NBA LSTM Lineup Forecaster", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    with open("production_bayesian_network.pkl", "rb") as f:
        return pickle.load(f), json.load(open("production_feature_info.json"))

def main():
    st.title("🧠 NBA LSTM Lineup Forecaster")
    st.markdown("**LSTM-Powered Lineup Efficiency Predictions • 70.57% Accuracy**")
    
    model, feature_info = load_model()
    infer = VariableElimination(model)
    
    # Input sliders
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.slider("🔮 LSTM Projection Strength", 0, 3, 2)
        p2 = st.slider("🎯 FG% Level", 0, 3, 2)
        p3 = st.slider("📈 Plus/Minus", 0, 3, 2)
    with col2:
        p4 = st.slider("⭐ Net Rating", 0, 3, 2)
        p5 = st.slider("🤝 Assist Form", 0, 3, 2)
    
    if st.button("🧠 Predict Lineup Efficiency", type="primary"):
        evidence = {
            'PROJECTION_STRENGTH_LEVEL': p1, 'FG_PCT_LEVEL': p2, 'PLUS_MINUS_LEVEL': p3,
            'LINEUP_NET_RATING_TALENT_LEVEL': p4, 'AVG_FORM_RATIO_AST_LEVEL': p5
        }
        
        result = infer.query(variables=['LINEUP_QUALITY_SCORE_LEVEL'], evidence=evidence)
        probs = result.values
        pred_class = np.argmax(probs)
        
        # Display results
        st.success(f"**Prediction:** {['Low', 'Medium', 'High'][pred_class]} Efficiency")
        st.write("**Probabilities:**")
        st.progress(probs[0], text=f"Low: {probs[0]*100:.1f}%")
        st.progress(probs[1], text=f"Medium: {probs[1]*100:.1f}%") 
        st.progress(probs[2], text=f"High: {probs[2]*100:.1f}%")

if __name__ == "__main__":
    main()
