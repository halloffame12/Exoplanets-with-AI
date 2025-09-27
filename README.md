# 🪐 Exoplanet Detection System

A comprehensive Python application for automatically detecting exoplanets using NASA's open-source datasets (Kepler, K2, TESS) with machine learning classification.

## 🌟 Features

- **Data Management**: Download and preprocess data from NASA Exoplanet Archive
- **Machine Learning**: Train Random Forest and Gradient Boosting models
- **Predictions**: Classify new exoplanet candidates as Confirmed, Candidate, or False Positive
- **Analytics**: Visualize model performance and feature importance
- **Web Interface**: User-friendly Streamlit application
- **Real-time Updates**: Retrain models with new data

## 📊 Supported Datasets

- **Kepler**: Primary Kepler mission data
- **K2**: Extended Kepler mission data  
- **TESS**: Transiting Exoplanet Survey Satellite data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📁 Project Structure

```
nasa-hackathon/
├── app.py                 # Main Streamlit application
├── model.py              # Machine learning model implementation
├── preprocess.py         # Data preprocessing and cleaning
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Usage Guide

### 1. Data Management
- **Download NASA Data**: Access Kepler, K2, and TESS datasets directly from NASA's Exoplanet Archive
- **Upload Custom Data**: Use your own CSV datasets
- **Data Preview**: View dataset statistics and column information

### 2. Model Training
- **Model Selection**: Choose between Random Forest and Gradient Boosting
- **Hyperparameter Tuning**: Enable automatic hyperparameter optimization
- **Cross-Validation**: Configure cross-validation settings
- **Performance Metrics**: View accuracy, precision, recall, and F1-score

### 3. Predictions
- **Single Prediction**: Enter individual exoplanet parameters for classification
- **Batch Prediction**: Upload CSV files for bulk classification
- **Confidence Scores**: View prediction probabilities for each class

### 4. Analytics
- **Feature Importance**: Understand which features are most important for classification
- **Model Performance**: Visualize training and test metrics
- **Data Visualizations**: Explore orbital period vs radius relationships and class distributions

## 🎯 Key Features

### Data Preprocessing
- Handles missing values with intelligent imputation
- Normalizes numerical features using StandardScaler
- Encodes categorical labels (Confirmed, Candidate, False Positive)
- Splits data into training and testing sets (80%-20%)

### Machine Learning Models
- **Random Forest**: Ensemble method with good performance on tabular data
- **Gradient Boosting**: Sequential boosting algorithm for improved accuracy
- **Hyperparameter Tuning**: GridSearchCV for optimal parameter selection
- **Cross-Validation**: K-fold validation for robust performance estimation

### Web Interface
- **Interactive Dashboard**: Multi-page Streamlit application
- **Real-time Updates**: Dynamic model training and prediction
- **Data Visualization**: Interactive plots using Plotly
- **Export Functionality**: Download predictions and model results

## 📈 Model Performance

The application provides comprehensive evaluation metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 🔬 Scientific Background

### Exoplanet Detection
Exoplanets are planets that orbit stars other than our Sun. The most successful method for detecting exoplanets is the **transit method**, which measures the dimming of a star when a planet passes in front of it.

### Key Features for Classification
- **Orbital Period**: Time for one complete orbit around the star
- **Planetary Radius**: Size of the planet relative to Earth
- **Transit Duration**: Length of time the planet blocks the star's light
- **Transit Depth**: Amount of light blocked during transit
- **Impact Parameter**: How close the transit passes to the center of the star
- **Stellar Properties**: Temperature, surface gravity, and metallicity of the host star

### Classification Categories
- **Confirmed**: Verified exoplanet with additional validation
- **Candidate**: Potential exoplanet requiring further study
- **False Positive**: Not an exoplanet (e.g., binary star system, instrumental noise)

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **requests**: HTTP library for data downloading
- **matplotlib/seaborn**: Data visualization
- **plotly**: Interactive visualizations
- **joblib**: Model persistence

### Data Sources
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **Kepler Data**: Primary mission exoplanet candidates
- **K2 Data**: Extended mission observations
- **TESS Data**: Current survey mission results

## 🚀 Advanced Features

### Hyperparameter Tuning
The application supports automatic hyperparameter optimization using GridSearchCV:
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, min_samples_split

### Model Persistence
- Save trained models to disk for later use
- Load previously trained models
- Export training data and predictions

### Batch Processing
- Upload multiple CSV files for prediction
- Process large datasets efficiently
- Export results in CSV format

## 🔍 Troubleshooting

### Common Issues

1. **Data Download Fails**
   - Check internet connection
   - Verify NASA Exoplanet Archive is accessible
   - Try using sample data instead

2. **Model Training Errors**
   - Ensure data has required columns
   - Check for sufficient data (minimum 100 samples recommended)
   - Verify target column has multiple classes

3. **Prediction Errors**
   - Ensure input data matches training features
   - Check data types and ranges
   - Verify model is trained before making predictions

### Performance Tips
- Use hyperparameter tuning for better accuracy (takes longer)
- Ensure sufficient training data (1000+ samples recommended)
- Consider feature engineering for domain-specific improvements

## 📚 References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/overview/index.html)
- [TESS Mission](https://tess.mit.edu/)
- [Exoplanet Detection Methods](https://exoplanets.nasa.gov/5-ways-to-find-a-planet/)

## 🤝 Contributing

This project was created for the NASA Hackathon. Contributions and improvements are welcome!

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Exoplanet Hunting! 🪐✨**
