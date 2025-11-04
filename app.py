import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np

# Import pgmpy - this should work with your version
try:
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
    st.success("✅ pgmpy 1.0.0 loaded successfully!")
except ImportError as e:
    PGMPY_AVAILABLE = False
    st.error(f"❌ pgmpy import failed: {e}")

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
        st.error(f"Model loading error: {e}")
        return None, None

def main():
    st.title("🧠 NBA LSTM Lineup Forecaster")
    st.markdown("**LSTM Neural Network + Bayesian Network • 70.57% Accuracy**")
    
    # Load model
    model, feature_info = load_model()
    
    if not PGMPY_AVAILABLE:
        st.error("pgmpy not available - cannot run Bayesian Network predictions")
        return
        
    if model is None:
        st.error("Failed to load AI model")
        return
        
    infer = VariableElimination(model)
    
    # Input interface
    st.header("📊 Lineup Efficiency Prediction")
    
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
            'PROJECTION_STRENGTH_LEVEL': p1, 
            'FG_PCT_LEVEL': p2, 
            'PLUS_MINUS_LEVEL': p3,
            'LINEUP_NET_RATING_TALENT_LEVEL': p4, 
            'AVG_FORM_RATIO_AST_LEVEL': p5
        }
        
        try:
            # Real Bayesian Network prediction
            result = infer.query(variables=['LINEUP_QUALITY_SCORE_LEVEL'], evidence=evidence)
            probs = result.values
            pred_class = np.argmax(probs)
            
            # Display results
            efficiency = ["Low", "Medium", "High"][pred_class]
            st.success(f"**🎯 Prediction:** {efficiency} Efficiency")
            st.metric("Confidence", f"{probs[pred_class]*100:.1f}%")
            
            # Probability distribution
            st.subheader("Probability Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low Efficiency", f"{probs[0]*100:.1f}%")
                st.progress(probs[0])
            with col2:
                st.metric("Medium Efficiency", f"{probs[1]*100:.1f}%")
                st.progress(probs[1])
            with col3:
                st.metric("High Efficiency", f"{probs[2]*100:.1f}%")
                st.progress(probs[2])
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Model info
    with st.expander("🔧 Model Information"):
        st.markdown(f"""
        **NBA Hybrid AI System**
        
        **Architecture:** LSTM Forecasting → Bayesian Network Inference
        **Accuracy:** 70.57% (+3.23% improvement)
        **pgmpy Version:** 1.0.0 ✅
        
        **Features Used:**
        - 🔮 LSTM Projection Strength
        - 🎯 Field Goal Percentage  
        - 📈 Plus/Minus Impact
        - ⭐ Net Rating Talent
        - 🤝 Assist Form Ratio
        
        **Technical Stack:**
        - Python 3.10
        - pgmpy 1.0.0 (Bayesian Networks)
        - pandas 2.2.2
        - numpy 2.0.2
        - scikit-learn 1.6.1
        """)

if __name__ == "__main__":
    main()
