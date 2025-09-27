"""
Streamlit web application for exoplanet detection using NASA datasets.
Provides an interactive interface for data upload, model training, and prediction.
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

# Import our custom modules
from preprocess import ExoplanetDataProcessor, create_sample_data
from model import ExoplanetClassifier, create_visualization_plots

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

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🪐 Exoplanet Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Detect exoplanets using NASA's Kepler, K2, and TESS datasets with machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Management", "🤖 Model Training", "🔮 Predictions", "📈 Analytics", "⚙️ Settings"]
    )
    
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
        
        This application uses machine learning to classify exoplanet candidates from NASA's open-source datasets:
        
        ### 🌟 Features
        - **Data Management**: Download and preprocess data from NASA Exoplanet Archive
        - **Machine Learning**: Train Random Forest and Gradient Boosting models
        - **Predictions**: Classify new exoplanet candidates as Confirmed, Candidate, or False Positive
        - **Analytics**: Visualize model performance and feature importance
        - **Real-time Updates**: Retrain models with new data
        
        ### 📊 Supported Datasets
        - **Kepler**: Primary Kepler mission data
        - **K2**: Extended Kepler mission data  
        - **TESS**: Transiting Exoplanet Survey Satellite data
        
        ### 🚀 Quick Start
        1. Go to **Data Management** to download NASA datasets
        2. Visit **Model Training** to train your classification model
        3. Use **Predictions** to classify new exoplanet candidates
        4. Check **Analytics** for model performance insights
        """)
    
    with col2:
        st.markdown("### 📈 System Status")
        
        # Model status
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
            if st.session_state.model:
                summary = st.session_state.model.get_model_summary()
                st.write(f"**Model Type:** {summary.get('model_type', 'Unknown')}")
                st.write(f"**Features:** {summary.get('feature_count', 0)}")
                st.write(f"**Classes:** {summary.get('class_count', 0)}")
        else:
            st.warning("⚠️ No Model Trained")
        
        # Data status
        if st.session_state.training_data is not None:
            st.success("✅ Training Data Loaded")
            st.write(f"**Records:** {len(st.session_state.training_data)}")
        else:
            st.info("ℹ️ No Training Data")
        
        # Quick actions
        st.markdown("### 🎯 Quick Actions")
        if st.button("Download Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data()
                st.session_state.training_data = sample_data
                st.success("Sample data generated!")
                st.rerun()
        
        if st.button("Train Default Model", use_container_width=True):
            if st.session_state.training_data is not None:
                with st.spinner("Training model..."):
                    train_default_model()
                st.success("Model trained successfully!")
                st.rerun()
            else:
                st.error("Please load training data first!")

def show_data_management_page():
    """Display the data management page."""
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Download NASA Data", "Upload Custom Data", "Data Preview"])
    
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
            if st.button("Download Dataset", use_container_width=True):
                with st.spinner(f"Downloading {dataset_type.upper()} data from NASA..."):
                    try:
                        df = st.session_state.processor.download_nasa_data(dataset_type)
                        if df is not None and len(df) > 0:
                            st.session_state.training_data = df
                            st.success(f"Successfully downloaded {len(df)} records!")
                        else:
                            st.error("Failed to download data. Please try again.")
                    except Exception as e:
                        st.error(f"Error downloading data: {str(e)}")
    
    with tab2:
        st.subheader("Upload Custom Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your own exoplanet dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.training_data = df
                st.success(f"Successfully uploaded {len(df)} records!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab3:
        st.subheader("Data Preview")
        
        if st.session_state.training_data is not None:
            df = st.session_state.training_data
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Data preview
            st.subheader("Data Sample")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
        else:
            st.info("No data loaded. Please download or upload a dataset first.")

def show_model_training_page():
    """Display the model training page."""
    st.header("🤖 Model Training")
    
    if st.session_state.training_data is None:
        st.warning("Please load training data first in the Data Management page.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting"],
            help="Choose the machine learning algorithm"
        )
        
        hyperparameter_tuning = st.checkbox(
            "Enable Hyperparameter Tuning",
            help="Automatically tune hyperparameters (takes longer)"
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
            value=42,
            help="Random seed for reproducibility"
        )
    
    # Training button
    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                train_model(model_type, hyperparameter_tuning, test_size/100, cv_folds, random_state)
                st.success("Model trained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Display current model info
    if st.session_state.model_trained and st.session_state.model:
        st.subheader("Current Model Information")
        
        summary = st.session_state.model.get_model_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", summary.get('model_type', 'Unknown').title())
        with col2:
            st.metric("Features", summary.get('feature_count', 0))
        with col3:
            st.metric("Classes", summary.get('class_count', 0))
        
        # Training metrics
        if 'training_metrics' in summary:
            st.subheader("Training Metrics")
            metrics = summary['training_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

def show_predictions_page():
    """Display the predictions page."""
    st.header("🔮 Exoplanet Predictions")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Single Observation Prediction")
        
        # Create input form
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
                # Prepare input data
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
                
                # Make prediction
                try:
                    prediction = st.session_state.model.predict(input_data)[0]
                    probabilities = st.session_state.model.predict_proba(input_data)[0]
                    
                    # Decode prediction
                    class_names = st.session_state.model.class_names
                    prediction_label = class_names[prediction]
                    
                    # Display results
                    st.success(f"Prediction: **{prediction_label}**")
                    
                    # Show probabilities
                    st.subheader("Prediction Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(prob_df, x='Class', y='Probability', 
                               title="Prediction Confidence",
                               color='Probability',
                               color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.subheader("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch prediction",
            type="csv",
            help="Upload a CSV file with exoplanet data for batch prediction"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records for prediction")
                
                if st.button("🔮 Predict All", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        # Preprocess the data
                        df_clean = st.session_state.processor.clean_data(df)
                        X, _, _ = st.session_state.processor.prepare_features(df_clean)
                        X_scaled = st.session_state.processor.transform(X)
                        
                        # Make predictions
                        predictions = st.session_state.model.predict(X_scaled)
                        probabilities = st.session_state.model.predict_proba(X_scaled)
                        
                        # Decode predictions
                        class_names = st.session_state.model.class_names
                        prediction_labels = [class_names[pred] for pred in predictions]
                        
                        # Add predictions to dataframe
                        df_with_predictions = df_clean.copy()
                        df_with_predictions['Prediction'] = prediction_labels
                        
                        for i, class_name in enumerate(class_names):
                            df_with_predictions[f'Prob_{class_name}'] = probabilities[:, i]
                        
                        st.session_state.predictions = df_with_predictions
                        
                        st.success(f"Predictions completed for {len(df)} records!")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df_with_predictions, use_container_width=True)
                        
                        # Download results
                        csv = df_with_predictions.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name=f"exoplanet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def show_analytics_page():
    """Display the analytics page."""
    st.header("📈 Model Analytics")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first to view analytics.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Performance", "Data Visualizations"])
    
    with tab1:
        st.subheader("Feature Importance")
        
        if st.session_state.model:
            importance_df = st.session_state.model.get_feature_importance()
            
            # Create bar chart
            fig = px.bar(importance_df, x='importance', y='feature', 
                        orientation='h', title="Feature Importance",
                        color='importance', color_continuous_scale='viridis')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.subheader("Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
    
    with tab2:
        st.subheader("Model Performance")
        
        if st.session_state.model:
            summary = st.session_state.model.get_model_summary()
            
            # Training metrics
            if 'training_metrics' in summary:
                metrics = summary['training_metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Metrics bar chart
                    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    metric_values = [metrics.get('accuracy', 0), metrics.get('precision', 0), 
                                   metrics.get('recall', 0), metrics.get('f1_score', 0)]
                    
                    fig = px.bar(x=metric_names, y=metric_values, 
                               title="Model Performance Metrics",
                               color=metric_values, color_continuous_scale='viridis')
                    fig.update_layout(yaxis_title="Score", xaxis_title="Metric")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics table
                    metrics_df = pd.DataFrame({
                        'Metric': metric_names,
                        'Score': metric_values
                    })
                    st.dataframe(metrics_df, use_container_width=True)
    
    with tab3:
        st.subheader("Data Visualizations")
        
        if st.session_state.training_data is not None:
            df = st.session_state.training_data
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Orbital period vs radius
                if 'period' in df.columns and 'radius' in df.columns:
                    fig = px.scatter(df, x='period', y='radius', 
                                   title="Orbital Period vs Planetary Radius",
                                   labels={'period': 'Orbital Period (days)', 
                                          'radius': 'Planetary Radius (Earth radii)'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Class distribution
                if 'status' in df.columns:
                    class_counts = df['status'].value_counts()
                    fig = px.pie(values=class_counts.values, names=class_counts.index,
                               title="Class Distribution")
                    st.plotly_chart(fig, use_container_width=True)

def show_settings_page():
    """Display the settings page."""
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Settings")
        
        # Model persistence
        if st.button("💾 Save Model"):
            if st.session_state.model_trained:
                try:
                    st.session_state.model.save_model("exoplanet_model.pkl")
                    st.success("Model saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
            else:
                st.warning("No trained model to save")
        
        if st.button("📁 Load Model"):
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
                st.error(f"Error loading model: {str(e)}")
    
    with col2:
        st.subheader("Data Settings")
        
        # Clear data
        if st.button("🗑️ Clear All Data"):
            st.session_state.training_data = None
            st.session_state.model = None
            st.session_state.model_trained = False
            st.session_state.predictions = None
            st.success("All data cleared!")
            st.rerun()
        
        # Export data
        if st.button("📤 Export Training Data"):
            if st.session_state.training_data is not None:
                csv = st.session_state.training_data.to_csv(index=False)
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
        # Use sample data if no training data
        if st.session_state.training_data is None:
            st.session_state.training_data = create_sample_data()
        
        # Preprocess data
        df_clean = st.session_state.processor.clean_data(st.session_state.training_data)
        X, y, features = st.session_state.processor.prepare_features(df_clean)
        X_train, X_test, y_train, y_test = st.session_state.processor.split_data(X, y)
        X_train_scaled, X_test_scaled = st.session_state.processor.fit_transform(X_train, X_test)
        y_train_encoded, y_test_encoded = st.session_state.processor.encode_labels(y_train, y_test)
        
        # Train model
        st.session_state.model = ExoplanetClassifier('random_forest')
        st.session_state.model.train(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)
        st.session_state.model_trained = True
        
    except Exception as e:
        st.error(f"Error training default model: {str(e)}")

def train_model(model_type, hyperparameter_tuning, test_size, cv_folds, random_state):
    """Train a model with specified parameters."""
    try:
        # Preprocess data
        df_clean = st.session_state.processor.clean_data(st.session_state.training_data)
        X, y, features = st.session_state.processor.prepare_features(df_clean)
        X_train, X_test, y_train, y_test = st.session_state.processor.split_data(X, y, test_size, random_state)
        X_train_scaled, X_test_scaled = st.session_state.processor.fit_transform(X_train, X_test)
        y_train_encoded, y_test_encoded = st.session_state.processor.encode_labels(y_train, y_test)
        
        # Train model
        st.session_state.model = ExoplanetClassifier(model_type)
        st.session_state.model.train(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
                                   hyperparameter_tuning, cv_folds)
        st.session_state.model_trained = True
        
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

if __name__ == "__main__":
    main()
