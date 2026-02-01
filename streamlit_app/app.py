"""
Streamlit Fraud Detection Prediction App
Loads random transactions and predicts fraud probability using MLflow model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from pathlib import Path
import json

from preprocessing import FraudPreprocessor


# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #ff4b4b;
        background-color: #ffe6e6;
        margin: 1rem 0;
    }
    .legit-box {
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #00cc00;
        background-color: #e6ffe6;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str = "fraud_detection_lgbm", stage: str = "latest"):
    """
    Load model from MLflow registry.
    Cached to avoid reloading on every interaction.
    """
    try:
        mlflow_uri = st.session_state.get('mlflow_uri', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Try to load from registry first
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            st.sidebar.success(f"✓ Loaded model from Production stage")
            return model, "Production"
        except:
            # Fallback to latest version
            model_uri = f"models:/{model_name}/None"
            model = mlflow.pyfunc.load_model(model_uri)
            st.sidebar.success(f"✓ Loaded latest model version")
            return model, "Latest"
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("💡 Make sure MLflow is running and model is registered")
        return None, None


@st.cache_resource
def load_preprocessor():
    """
    Load the preprocessing pipeline.
    Cached to avoid reloading artifacts.
    """
    try:
        preprocessor = FraudPreprocessor()
        st.sidebar.success(f"✓ Loaded preprocessor ({len(preprocessor.feature_names)} features)")
        return preprocessor
    except Exception as e:
        st.error(f"❌ Failed to load preprocessor: {e}")
        st.info("💡 Make sure preprocess_artifacts/ contains all required files")
        return None


@st.cache_data
def load_sample_transactions():
    """
    Load sample transactions dataset.
    """
    try:
        df = pd.read_csv("sample_data/raw_transactions.csv")
        st.sidebar.success(f"✓ Loaded {len(df)} sample transactions")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")
        return None


def get_random_transaction(df: pd.DataFrame) -> pd.Series:
    """
    Get a random transaction from the dataset.
    """
    return df.sample(n=1).iloc[0]


def display_transaction_details(tx: pd.Series):
    """
    Display key transaction information.
    """
    st.subheader("📋 Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Key Information")
        st.markdown(f"**Transaction ID:** {tx.get('TransactionID', 'N/A')}")
        st.markdown(f"**Amount:** ${tx.get('TransactionAmt', 0):.2f}")
        st.markdown(f"**Product CD:** {tx.get('ProductCD', 'N/A')}")
        
    with col2:
        st.markdown("##### Card Information")
        st.markdown(f"**Card1:** {tx.get('card1', 'N/A')}")
        st.markdown(f"**Card2:** {tx.get('card2', 'N/A')}")
        st.markdown(f"**Card3:** {tx.get('card3', 'N/A')}")
        st.markdown(f"**Card4:** {tx.get('card4', 'N/A')}")
        st.markdown(f"**Card5:** {tx.get('card5', 'N/A')}")
        st.markdown(f"**Card6:** {tx.get('card6', 'N/A')}")
        
    with col3:
        st.markdown("##### Email & Address")
        st.markdown(f"**P Email Domain:** {tx.get('P_emaildomain', 'N/A')}")
        st.markdown(f"**R Email Domain:** {tx.get('R_emaildomain', 'N/A')}")
        st.markdown(f"**Address 1:** {tx.get('addr1', 'N/A')}")
        st.markdown(f"**Address 2:** {tx.get('addr2', 'N/A')}")


def display_prediction_result(fraud_prob: float, actual_label: int = None):
    """
    Display prediction result with visual styling.
    """
    st.subheader("🎯 Prediction Result")
    
    is_fraud = fraud_prob > 0.5
    threshold = 0.5
    
    # Prediction display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_fraud:
            st.markdown(f"""
                <div class="fraud-box">
                    <h2 style="color: #ff4b4b; margin: 0;">⚠️ FRAUD DETECTED</h2>
                    <h3 style="margin-top: 1rem;">Fraud Probability: {fraud_prob*100:.2f}%</h3>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        This transaction is highly suspicious and should be flagged for review.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="legit-box">
                    <h2 style="color: #00cc00; margin: 0;">✅ LEGITIMATE</h2>
                    <h3 style="margin-top: 1rem;">Fraud Probability: {fraud_prob*100:.2f}%</h3>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        This transaction appears normal and can proceed.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Confidence gauge
        st.markdown("### Confidence")
        confidence = abs(fraud_prob - 0.5) * 2  # 0 at boundary, 1 at extremes
        st.progress(confidence)
        st.markdown(f"**{confidence*100:.1f}%** confidence")
        
        # Risk level
        st.markdown("### Risk Level")
        if fraud_prob > 0.8:
            st.error("🔴 HIGH RISK")
        elif fraud_prob > 0.5:
            st.warning("🟡 MEDIUM RISK")
        elif fraud_prob > 0.2:
            st.info("🔵 LOW RISK")
        else:
            st.success("🟢 MINIMAL RISK")
    
    # Ground truth comparison (if available)
    if actual_label is not None:
        st.markdown("---")
        st.subheader("📊 Ground Truth Comparison")
        
        actual_text = "FRAUD" if actual_label == 1 else "LEGITIMATE"
        prediction_text = "FRAUD" if is_fraud else "LEGITIMATE"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Label", actual_text)
        with col2:
            st.metric("Prediction", prediction_text)
        with col3:
            is_correct = (actual_label == 1) == is_fraud
            accuracy_text = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            st.metric("Result", accuracy_text)


def main():
    """
    Main Streamlit app.
    """
    # Header
    st.markdown('<div class="main-header">🔍 Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # MLflow URI configuration
    mlflow_uri_input = st.sidebar.text_input(
        "MLflow Tracking URI",
        value=st.session_state.get('mlflow_uri', 'http://mlflow:5000'),
        help="URI of the MLflow tracking server"
    )
    
    if mlflow_uri_input != st.session_state.get('mlflow_uri', ''):
        st.session_state['mlflow_uri'] = mlflow_uri_input
        st.sidebar.info("Click 'Reload Model' to apply changes")
    
    # Model selection
    model_name = st.sidebar.text_input(
        "Model Name",
        value="fraud_detection_lgbm",
        help="Name of the registered model in MLflow"
    )
    
    # Reload button
    if st.sidebar.button("🔄 Reload Model & Preprocessor"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Load components
    st.sidebar.markdown("### 📦 System Status")
    
    with st.spinner("Loading components..."):
        samples_df = load_sample_transactions()
        preprocessor = load_preprocessor()
        model, model_stage = load_model(model_name)
    
    # Check if all components loaded successfully
    if samples_df is None or preprocessor is None or model is None:
        st.error("❌ System initialization failed. Please check the sidebar for details.")
        st.stop()
    
    st.sidebar.markdown(f"**Model Stage:** {model_stage}")
    st.sidebar.markdown(f"**Features:** {len(preprocessor.feature_names)}")
    st.sidebar.markdown(f"**Sample Size:** {len(samples_df)}")
    
    # Main interface
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎲 Select a Transaction to Analyze")
        st.markdown("Click the button below to randomly select a transaction and predict its fraud probability.")
    
    with col2:
        predict_button = st.button("🎯 Get Random Transaction & Predict", use_container_width=True)
    
    # Prediction logic
    if predict_button or 'current_transaction' in st.session_state:
        
        if predict_button:
            # Get new random transaction
            st.session_state['current_transaction'] = get_random_transaction(samples_df)
        
        tx = st.session_state['current_transaction']
        
        st.markdown("---")
        
        # Display transaction details
        display_transaction_details(tx)
        
        st.markdown("---")
        
        # Preprocess and predict
        with st.spinner("🔄 Processing transaction..."):
            try:
                # Preprocess
                feature_vector = preprocessor.preprocess(tx)
                
                # Predict
                fraud_prob = model.predict(feature_vector)[0]
                
                # Display result
                actual_label = int(tx.get('isFraud', -1)) if 'isFraud' in tx.index else None
                display_prediction_result(fraud_prob, actual_label)
                
                # Additional info expander
                with st.expander("🔧 Advanced Details"):
                    st.markdown("#### Feature Vector Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Features", feature_vector.shape[1])
                    with col2:
                        st.metric("Non-null Features", np.sum(~np.isnan(feature_vector)))
                    with col3:
                        null_pct = np.sum(np.isnan(feature_vector)) / feature_vector.shape[1] * 100
                        st.metric("Null %", f"{null_pct:.1f}%")
                    
                    st.markdown("#### Raw Prediction Score")
                    st.code(f"Fraud Probability: {fraud_prob:.6f}")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)
    
    else:
        # Initial state - show instructions
        st.info("👆 Click the button above to start analyzing transactions!")
        
        # Show some stats about the dataset
        st.markdown("---")
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(samples_df))
        
        with col2:
            if 'isFraud' in samples_df.columns:
                fraud_count = samples_df['isFraud'].sum()
                st.metric("Fraud Cases", fraud_count)
        
        with col3:
            if 'isFraud' in samples_df.columns:
                fraud_pct = (samples_df['isFraud'].sum() / len(samples_df)) * 100
                st.metric("Fraud Rate", f"{fraud_pct:.1f}%")
        
        with col4:
            st.metric("Total Features", len(samples_df.columns))


if __name__ == "__main__":
    main()
