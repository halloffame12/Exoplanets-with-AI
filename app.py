"""
Streamlit web application for exoplanet detection using NASA datasets.
Provides an interactive interface for data upload, model training, prediction, and light curve analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import shap
from loguru import logger
import yaml
from preprocess import ExoplanetDataProcessor, create_sample_data, ensure_streamlit_compatibility
from model import ExoplanetClassifier, create_visualization_plots

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

logger.add("logs/app.log", rotation="500 MB")

# Page configuration
st.set_page_config(
    page_title="Exoplanet Detection System",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'processor' not in st.session_state:
        st.session_state.processor = ExoplanetDataProcessor()
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'reference_data' not in st.session_state:
        st.session_state.reference_data = None
    if 'global_seed' not in st.session_state:
        st.session_state.global_seed = CONFIG['app']['global_seed']
        np.random.seed(st.session_state.global_seed)

def main():
    """Main application function."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🪐 Exoplanet Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Detect exoplanets using NASA's Kepler, K2, and TESS datasets with advanced machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Management", "🤖 Model Training", "🔮 Predictions", "📈 Analytics", "⚙️ Settings"]
    )
    
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
    st.markdown(f'<style>:root {{ --background-color: {"#fff" if theme == "Light" else "#333"}; }}</style>', unsafe_allow_html=True)
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Management":
        show_data_management_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page with project overview."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Exoplanet Detection System
        
        This application leverages machine learning and deep learning to classify exoplanet candidates using NASA's open-source datasets, including light curve analysis for enhanced accuracy.
        
        ### 🌟 Features
        - **Data Management**: Download NASA datasets or upload custom data
        - **Light Curve Analysis**: Extract features from light curves using `lightkurve`
        - **Machine Learning**: Train Random Forest, Gradient Boosting, XGBoost, and Neural Network models
        - **Predictions**: Classify exoplanet candidates as Confirmed, Candidate, or False Positive
        - **Analytics**: Visualize model performance, feature importance, and data distributions
        - **Real-time Updates**: Retrain models with new data
        
        ### 📊 Supported Datasets
        - **Kepler**: Primary Kepler mission data
        - **K2**: Extended Kepler mission data  
        - **TESS**: Transiting Exoplanet Survey Satellite data
        
        ### 🚀 Quick Start
        1. Go to **Data Management** to download NASA datasets or upload custom data
        2. Visit **Model Training** to train your classification model
        3. Use **Predictions** to classify new exoplanet candidates
        4. Check **Analytics** for model performance insights
        """)
    
    with col2:
        st.markdown("### 📈 System Status")
        
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
            if st.session_state.model:
                summary = st.session_state.model.get_model_summary()
                st.write(f"**Model Type:** {summary.get('model_type', 'Unknown').title()}")
                st.write(f"**Features:** {summary.get('feature_count', 0)}")
                st.write(f"**Classes:** {summary.get('class_count', 0)}")
        else:
            st.warning("⚠️ No Model Trained")
        
        if st.session_state.training_data is not None:
            st.success("✅ Training Data Loaded")
            st.write(f"**Records:** {len(st.session_state.training_data)}")
        else:
            st.info("ℹ️ No Training Data")
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("Download Sample Data", key="quick_sample"):
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data()
                st.session_state.training_data = sample_data
                st.success("Sample data generated!")
                st.rerun()
        
        if st.button("Train Default Model", key="quick_train"):
            if st.session_state.training_data is not None:
                with st.spinner("Training model..."):
                    train_default_model()
                st.success("Model trained successfully!")
                st.rerun()
            else:
                st.error("Please load training data first!")

def show_data_management_page():
    """Display the data management page with light curve processing."""
    st.header("📊 Data Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Download NASA Data", "Upload Custom Data", "Light Curve Analysis", "Data Preview"])
    
    with tab1:
        st.subheader("Download from NASA Exoplanet Archive")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            dataset_type = st.selectbox(
                "Select Dataset",
                ["kepler", "k2", "tess"],
                help="Choose which NASA dataset to download"
            )
        
        with col2:
            if st.button("Download Dataset", key="download_nasa"):
                with st.spinner(f"Downloading {dataset_type.upper()} data from NASA..."):
                    try:
                        df = st.session_state.processor.download_nasa_data(dataset_type)
                        if df is not None and len(df) > 0:
                            st.session_state.training_data = df
                            st.success(f"Successfully downloaded {len(df)} records!")
                        else:
                            st.error("Failed to download data. Please try again.")
                    except Exception as e:
                        logger.error(f"Error downloading data: {str(e)}")
                        st.error(f"Error downloading data: {str(e)}")
                        st.info("Please try again or check the NASA Exoplanet Archive status.")
    
    with tab2:
        st.subheader("Upload Custom Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your own exoplanet dataset in CSV format"
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > CONFIG['app']['upload_max_size_mb'] * 1024 * 1024:
                st.error("File too large!")
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df) == 0:
                        st.error("The uploaded file is empty!")
                    elif len(df.columns) < 3:
                        st.error("The uploaded file doesn't have enough columns. Please ensure it has at least 3 columns.")
                    else:
                        df = ensure_streamlit_compatibility(df)
                        st.session_state.training_data = df
                        st.success(f"Successfully uploaded {len(df)} records with {len(df.columns)} columns!")
                except Exception as e:
                    logger.error(f"Error reading file: {str(e)}")
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please ensure the file is a valid CSV with proper formatting.")
    
    with tab3:
        st.subheader("Light Curve Analysis")
        
        target_name = st.text_input("Target Name (e.g., Kepler-10)", value="Kepler-10")
        mission = st.selectbox("Mission", ["Kepler", "K2", "TESS"])
        
        if st.button("Analyze Light Curve", key="analyze_light_curve"):
            with st.spinner(f"Processing light curve for {target_name}..."):
                try:
                    features = st.session_state.processor.process_light_curve(target_name, mission=mission)
                    if features is not None:
                        st.success("Light curve processed successfully!")
                        features_df = pd.DataFrame([features])
                        st.dataframe(features_df, use_container_width=True)
                        
                        # Visualize periodogram
                        import lightkurve as lk
                        search_result = lk.search_lightcurve(target_name, mission=mission)
                        if len(search_result) > 0:
                            lc = search_result.download().normalize().remove_outliers()
                            pg = lc.to_periodogram()
                            fig = pg.plot()
                            st.pyplot(fig)
                    else:
                        st.error("Failed to process light curve. Please check the target name or mission.")
                except Exception as e:
                    logger.error(f"Error processing light curve: {str(e)}")
                    st.error(f"Error processing light curve: {str(e)}")
    
    with tab4:
        st.subheader("Data Preview")
        
        if st.session_state.training_data is not None:
            df = st.session_state.training_data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.subheader("Data Sample")
            df_display = ensure_streamlit_compatibility(df.head(10))
            st.dataframe(df_display, use_container_width=True)
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            col_info_display = ensure_streamlit_compatibility(col_info)
            st.dataframe(col_info_display, use_container_width=True)
            
        else:
            st.info("No data loaded. Please download or upload a dataset first.")

def show_model_training_page():
    """Display the model training page with neural network support."""
    st.header("🤖 Model Training")
    
    if st.session_state.training_data is None:
        st.warning("Please load training data first in the Data Management page.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting", "xgboost", "neural_net"],
            help="Choose the machine learning algorithm"
        )
        
        hyperparameter_tuning = st.checkbox(
            "Enable Hyperparameter Tuning",
            help="Automatically tune hyperparameters (takes longer, not available for neural network)"
        )
        
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data to use for testing"
        )
    
    with col2:
        st.subheader("Training Options")
        
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for cross-validation"
        )
        
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=CONFIG['app']['random_state'],
            help="Random seed for reproducibility"
        )
        
        if model_type == "neural_net":
            epochs = st.number_input(
                "Epochs",
                min_value=10,
                max_value=200,
                value=CONFIG['model']['default_hyperparams']['neural_net']['epochs'],
                help="Number of training epochs for neural network"
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=8,
                max_value=128,
                value=CONFIG['model']['default_hyperparams']['neural_net']['batch_size'],
                help="Batch size for neural network training"
            )
    
    if st.button("🚀 Train Model", key="train_model", use_container_width=True, type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                if st.session_state.training_data is None:
                    st.error("No training data available. Please load data first.")
                elif len(st.session_state.training_data) < 50:
                    st.error("Insufficient data for training. Please provide at least 50 samples.")
                else:
                    if model_type == "neural_net":
                        CONFIG['model']['default_hyperparams']['neural_net']['epochs'] = epochs
                        CONFIG['model']['default_hyperparams']['neural_net']['batch_size'] = batch_size
                    train_model(model_type, hyperparameter_tuning, test_size/100, cv_folds, random_state)
                    st.session_state.reference_data = st.session_state.training_data
                    st.success("Model trained successfully!")
                    st.rerun()
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                st.error(f"Error training model: {str(e)}")
                st.info("Please check your data and try again. Ensure all required columns are present.")
    
    if st.session_state.model_trained and st.session_state.model:
        st.subheader("✅ Model Successfully Trained!")
        
        summary = st.session_state.model.get_model_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", summary.get('model_type', 'Unknown').title())
        with col2:
            st.metric("Features", summary.get('feature_count', 0))
        with col3:
            st.metric("Classes", summary.get('class_count', 0))
        
        if st.session_state.training_results:
            st.subheader("📊 Training Performance")
            train_metrics = st.session_state.training_results['train_metrics']
            test_metrics = st.session_state.training_results['test_metrics']
            
            st.markdown("**Training Set Performance:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{train_metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{train_metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{train_metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{train_metrics.get('f1_score', 0):.4f}")
            
            st.markdown("**Test Set Performance:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{test_metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{test_metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{test_metrics.get('f1_score', 0):.4f}")
            
            test_acc = test_metrics.get('accuracy', 0)
            if test_acc > 0.8:
                st.success("🎉 Excellent model performance!")
            elif test_acc > 0.7:
                st.info("✅ Good model performance")
            elif test_acc > 0.6:
                st.warning("⚠️ Moderate model performance - consider hyperparameter tuning")
            else:
                st.error("❌ Poor model performance - check your data quality")

def show_predictions_page():
    """Display the predictions page with light curve input option."""
    st.header("🔮 Exoplanet Predictions")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Light Curve Prediction"])
    
    with tab1:
        st.subheader("Single Observation Prediction")
        
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                period = st.number_input("Orbital Period (days)", min_value=0.1, value=10.0, step=0.1)
                radius = st.number_input("Planetary Radius (Earth radii)", min_value=0.1, value=1.0, step=0.1)
                duration = st.number_input("Transit Duration (hours)", min_value=0.1, value=5.0, step=0.1)
                depth = st.number_input("Transit Depth (ppm)", min_value=0.1, value=100.0, step=1.0)
            
            with col2:
                impact = st.number_input("Impact Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                teff = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5500, step=100)
                logg = st.number_input("Stellar Surface Gravity (log g)", min_value=3.0, max_value=5.0, value=4.5, step=0.1)
                feh = st.number_input("Stellar Metallicity [Fe/H]", min_value=-2.0, max_value=1.0, value=0.0, step=0.1)
            
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
            
            if submitted:
                input_data = pd.DataFrame({
                    'period': [period],
                    'radius': [radius],
                    'duration': [duration],
                    'depth': [depth],
                    'impact': [impact],
                    'teff': [teff],
                    'logg': [logg],
                    'feh': [feh]
                })
                
                try:
                    if any(pd.isna(input_data.iloc[0])):
                        st.error("Please fill in all required fields with valid numbers.")
                    else:
                        input_scaled = st.session_state.processor.transform(input_data)
                        prediction = st.session_state.model.predict(input_scaled)[0]
                        probabilities = st.session_state.model.predict_proba(input_scaled)[0]
                        
                        class_names = st.session_state.model.class_names
                        prediction_label = class_names[prediction]
                        
                        st.success(f"Prediction: **{prediction_label}**")
                        
                        st.subheader("Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   title="Prediction Confidence",
                                   color='Probability',
                                   color_continuous_scale=CONFIG['plot']['style'])
                        fig.update_layout(meta={"alt": "Prediction confidence bar chart"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Prediction Explanation")
                        shap_values = st.session_state.model.explain(input_scaled)
                        if st.session_state.model.model_type == "neural_net":
                            explainer = shap.DeepExplainer(st.session_state.model.model, 
                                                         torch.tensor(input_scaled.values[:100], dtype=torch.float32).to(st.session_state.model.device))
                            shap_values = explainer.shap_values(torch.tensor(input_scaled.values, dtype=torch.float32).to(st.session_state.model.device))
                            shap_values = shap_values[prediction]
                        else:
                            explainer = shap.TreeExplainer(st.session_state.model.model)
                            shap_values = shap_values[prediction]
                        
                        shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                             base_values=explainer.expected_value[prediction], 
                                                             data=input_scaled.iloc[0], 
                                                             feature_names=input_scaled.columns))
                        st.pyplot(plt.gcf())
                    
                except Exception as e:
                    logger.error(f"Error making prediction: {str(e)}")
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please ensure all input values are valid numbers.")
    
    with tab2:
        st.subheader("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch prediction",
            type="csv",
            help="Upload a CSV file with exoplanet data for batch prediction"
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > CONFIG['app']['upload_max_size_mb'] * 1024 * 1024:
                st.error("File too large!")
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(df)} records for prediction")
                    
                    if st.button("🔮 Predict All", key="batch_predict", use_container_width=True):
                        with st.spinner("Making predictions..."):
                            try:
                                df_clean = st.session_state.processor.clean_data(df)
                                X, _, _ = st.session_state.processor.prepare_features(df_clean)
                                if len(X) == 0:
                                    st.error("No valid data found for prediction. Please check your data format.")
                                else:
                                    X_scaled = st.session_state.processor.transform(X)
                                    predictions = st.session_state.model.predict(X_scaled)
                                    probabilities = st.session_state.model.predict_proba(X_scaled)
                                    
                                    class_names = st.session_state.model.class_names
                                    prediction_labels = [class_names[pred] for pred in predictions]
                                    
                                    df_with_predictions = df_clean.copy()
                                    df_with_predictions['Prediction'] = prediction_labels
                                    
                                    for i, class_name in enumerate(class_names):
                                        df_with_predictions[f'Prob_{class_name}'] = probabilities[:, i]
                                    
                                    st.session_state.predictions = df_with_predictions
                                    
                                    st.success(f"Predictions completed for {len(df)} records!")
                                    
                                    st.subheader("Prediction Results")
                                    df_predictions_display = ensure_streamlit_compatibility(df_with_predictions)
                                    st.dataframe(df_predictions_display, use_container_width=True)
                                    
                                    csv = df_with_predictions.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Download Predictions",
                                        data=csv,
                                        file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            except Exception as e:
                                logger.error(f"Error processing predictions: {str(e)}")
                                st.error(f"Error processing predictions: {str(e)}")
                                st.info("Please ensure your data has the required columns and valid values.")
                            
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please ensure the file is a valid CSV with proper formatting.")
    
    with tab3:
        st.subheader("Light Curve Prediction")
        
        target_name = st.text_input("Target Name (e.g., Kepler-10)", value="Kepler-10", key="lc_target")
        mission = st.selectbox("Mission", ["Kepler", "K2", "TESS"], key="lc_mission")
        
        if st.button("🔮 Predict from Light Curve", key="lc_predict"):
            with st.spinner(f"Processing light curve for {target_name}..."):
                try:
                    features = st.session_state.processor.process_light_curve(target_name, mission=mission)
                    if features is not None:
                        input_data = pd.DataFrame([features])
                        input_scaled = st.session_state.processor.transform(input_data)
                        prediction = st.session_state.model.predict(input_scaled)[0]
                        probabilities = st.session_state.model.predict_proba(input_scaled)[0]
                        
                        class_names = st.session_state.model.class_names
                        prediction_label = class_names[prediction]
                        
                        st.success(f"Prediction: **{prediction_label}**")
                        
                        st.subheader("Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': class_names,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   title="Prediction Confidence",
                                   color='Probability',
                                   color_continuous_scale=CONFIG['plot']['style'])
                        fig.update_layout(meta={"alt": "Prediction confidence bar chart"})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to process light curve. Please check the target name or mission.")
                except Exception as e:
                    logger.error(f"Error predicting from light curve: {str(e)}")
                    st.error(f"Error predicting from light curve: {str(e)}")

def show_analytics_page():
    """Display the analytics page with enhanced visualizations."""
    st.header("📈 Model Analytics")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first to view analytics.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Model Performance", "Data Visualizations", "Drift Check"])
    
    with tab1:
        st.subheader("Feature Importance")
        
        if st.session_state.model:
            importance_df = st.session_state.model.get_feature_importance()
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', title="Feature Importance",
                        color='importance', color_continuous_scale=CONFIG['plot']['style'])
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, meta={"alt": "Feature importance bar chart"})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Importance Table")
            importance_display = ensure_streamlit_compatibility(importance_df)
            st.dataframe(importance_display, use_container_width=True)
    
    with tab2:
        st.subheader("Model Performance")
        
        if st.session_state.model:
            create_visualization_plots(st.session_state.model, 
                                    st.session_state.training_results['test_features'], 
                                    st.session_state.training_results['test_labels'])
    
    with tab3:
        st.subheader("Data Visualizations")
        
        if st.session_state.training_data is not None:
            df = st.session_state.training_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'period' in df.columns and 'radius' in df.columns:
                    fig = px.scatter(df, x='period', y='radius', 
                                   title="Orbital Period vs Planetary Radius",
                                   labels={'period': 'Orbital Period (days)', 
                                          'radius': 'Planetary Radius (Earth radii)'},
                                   color='status' if 'status' in df.columns else None,
                                   color_continuous_scale=CONFIG['plot']['style'])
                    fig.update_layout(meta={"alt": "Scatter plot of orbital period vs planetary radius"})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'status' in df.columns:
                    class_counts = df['status'].value_counts()
                    fig = px.pie(values=class_counts.values, names=class_counts.index,
                               title="Class Distribution",
                               color_discrete_sequence=px.colors.sequential.Viridis)
                    fig.update_layout(meta={"alt": "Pie chart of class distribution"})
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data Drift Check")
        
        if st.session_state.reference_data is None:
            st.warning("No reference data set. Train a model first to set reference.")
        else:
            new_data = st.file_uploader("Upload new data for drift check", type="csv")
            if new_data:
                try:
                    new_df = pd.read_csv(new_data)
                    drift_scores = {}
                    for col in st.session_state.reference_data.columns:
                        if col in new_df.columns and pd.api.types.is_numeric_dtype(st.session_state.reference_data[col]):
                            stat, p = ks_2samp(st.session_state.reference_data[col].dropna(), new_df[col].dropna())
                            drift_scores[col] = p
                    st.write("Drift p-values (low = significant drift):", drift_scores)
                    
                    if drift_scores:
                        drift_df = pd.DataFrame({'Column': list(drift_scores.keys()), 'p-value': list(drift_scores.values())})
                        fig = px.bar(drift_df, x='Column', y='p-value', 
                                   title="Data Drift Scores",
                                   color='p-value', color_continuous_scale=CONFIG['plot']['style'])
                        fig.update_layout(meta={"alt": "Bar chart of data drift p-values"})
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error in drift check: {str(e)}")
                    st.error(f"Error in drift check: {str(e)}")

def show_settings_page():
    """Display the settings page."""
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Settings")
        
        if st.button("💾 Save Model", key="save_model"):
            if st.session_state.model_trained:
                try:
                    st.session_state.model.save_model("exoplanet_model.pkl")
                    st.success("Model saved successfully!")
                except Exception as e:
                    logger.error(f"Error saving model: {str(e)}")
                    st.error(f"Error saving model: {str(e)}")
            else:
                st.warning("No trained model to save")
        
        if st.button("📁 Load Model", key="load_model"):
            try:
                if os.path.exists("exoplanet_model.pkl"):
                    st.session_state.model = ExoplanetClassifier()
                    st.session_state.model.load_model("exoplanet_model.pkl")
                    st.session_state.model_trained = True
                    st.success("Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("No saved model found")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                st.error(f"Error loading model: {str(e)}")
    
    with col2:
        st.subheader("Data Settings")
        
        if st.button("🗑️ Clear All Data", key="clear_data"):
            st.session_state.training_data = None
            st.session_state.model = None
            st.session_state.model_trained = False
            st.session_state.predictions = None
            st.session_state.reference_data = None
            st.session_state.training_results = None
            st.success("All data cleared!")
            st.rerun()
        
        if st.button("📤 Export Training Data", key="export_data"):
            if st.session_state.training_data is not None:
                csv = st.session_state.training_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Training Data",
                    data=csv,
                    file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No training data to export")

def train_default_model():
    """Train a default model with current data."""
    try:
        if st.session_state.training_data is None:
            st.session_state.training_data = create_sample_data()
        
        df_clean = st.session_state.processor.clean_data(st.session_state.training_data)
        X, y, features = st.session_state.processor.prepare_features(df_clean)
        X_train, X_test, y_train, y_test = st.session_state.processor.split_data(X, y)
        X_train_scaled, X_test_scaled = st.session_state.processor.fit_transform(X_train, X_test)
        y_train_encoded, y_test_encoded = st.session_state.processor.encode_labels(y_train, y_test)
        
        st.session_state.model = ExoplanetClassifier('random_forest')
        results = st.session_state.model.train(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)
        st.session_state.model_trained = True
        
        st.session_state.training_results = {
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'test_features': X_test_scaled,
            'test_labels': y_test_encoded
        }
        
        logger.info(f"Model trained successfully! Training accuracy: {results['train_metrics']['accuracy']:.4f}")
        logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error training default model: {str(e)}")
        st.error(f"Error training default model: {str(e)}")

def train_model(model_type, hyperparameter_tuning, test_size, cv_folds, random_state):
    """Train a model with specified parameters."""
    try:
        df_clean = st.session_state.processor.clean_data(st.session_state.training_data)
        X, y, features = st.session_state.processor.prepare_features(df_clean)
        X_train, X_test, y_train, y_test = st.session_state.processor.split_data(X, y, test_size, random_state)
        X_train_scaled, X_test_scaled = st.session_state.processor.fit_transform(X_train, X_test)
        y_train_encoded, y_test_encoded = st.session_state.processor.encode_labels(y_train, y_test)
        
        st.session_state.model = ExoplanetClassifier(model_type)
        results = st.session_state.model.train(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
                                   hyperparameter_tuning, cv_folds)
        st.session_state.model_trained = True
        
        st.session_state.training_results = {
            'train_metrics': results['train_metrics'],
            'test_metrics': results['test_metrics'],
            'test_features': X_test_scaled,
            'test_labels': y_test_encoded
        }
        
        logger.info(f"Model trained successfully! Training accuracy: {results['train_metrics']['accuracy']:.4f}")
        logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        st.error(f"Error training model: {str(e)}")
        raise Exception(f"Error training model: {str(e)}")

if __name__ == "__main__":
    main()