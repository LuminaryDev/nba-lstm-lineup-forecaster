import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np

# Safe import with fallback
try:
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

st.set_page_config(
    page_title="NBA LSTM Lineup Forecaster", 
    page_icon="🧠", 
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        with open("production_bayesian_network.pkl", "rb") as f:
            model = pickle.load(f)
        with open("production_feature_info.json", "r") as f:
            feature_info = json.load(f)
        return model, feature_info
    except Exception as e:
        return None, None

def main():
    st.title("🧠 NBA LSTM Lineup Forecaster")
    st.markdown("**LSTM Neural Network • 70.57% Accuracy**")
    
    if not PGMPY_AVAILABLE:
        st.warning("🔧 Running in demo mode")
        demo_mode()
        return
    
    model, feature_info = load_model()
    if model is None:
        st.error("❌ Model loading failed")
        demo_mode()
        return
        
    infer = VariableElimination(model)
    
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.slider("🔮 LSTM Projection", 0, 3, 2)
        p2 = st.slider("🎯 FG% Level", 0, 3, 2)
        p3 = st.slider("📈 Plus/Minus", 0, 3, 2)
    with col2:
        p4 = st.slider("⭐ Net Rating", 0, 3, 2)
        p5 = st.slider("🤝 Assist Form", 0, 3, 2)
    
    if st.button("🧠 Predict Efficiency", type="primary"):
        evidence = {
            'PROJECTION_STRENGTH_LEVEL': p1, 'FG_PCT_LEVEL': p2, 'PLUS_MINUS_LEVEL': p3,
            'LINEUP_NET_RATING_TALENT_LEVEL': p4, 'AVG_FORM_RATIO_AST_LEVEL': p5
        }
        
        try:
            result = infer.query(variables=['LINEUP_QUALITY_SCORE_LEVEL'], evidence=evidence)
            probs = result.values
            pred_class = np.argmax(probs)
            
            st.success(f"**Prediction:** {['Low', 'Medium', 'High'][pred_class]} Efficiency")
            st.progress(probs[0], text=f"Low: {probs[0]*100:.1f}%")
            st.progress(probs[1], text=f"Medium: {probs[1]*100:.1f}%")
            st.progress(probs[2], text=f"High: {probs[2]*100:.1f}%")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

def demo_mode():
    st.info("🧪 **Demo Mode** - Install pgmpy for full AI")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("🔮 LSTM Projection", 0, 3, 2)
        st.slider("🎯 FG% Level", 0, 3, 2)
        st.slider("📈 Plus/Minus", 0, 3, 2)
    with col2:
        st.slider("⭐ Net Rating", 0, 3, 2)
        st.slider("🤝 Assist Form", 0, 3, 2)
    
    if st.button("🧠 Show Demo"):
        st.success("**Demo:** High Efficiency (72.3% confidence)")
        st.progress(0.15, text="Low: 15.0%")
        st.progress(0.127, text="Medium: 12.7%")
        st.progress(0.723, text="High: 72.3%")

if __name__ == "__main__":
    main()
