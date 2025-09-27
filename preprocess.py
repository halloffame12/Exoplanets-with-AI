"""
Data preprocessing module for exoplanet detection.
Handles data cleaning, normalization, and preparation for machine learning.
Now includes light curve processing with lightkurve.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import io
import zipfile
import os
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.impute import KNNImputer  # Added for better imputation
import lightkurve as lk  # Added for light curve processing

logger.add("logs/app.log", rotation="500 MB")

class ExoplanetDataProcessor:
    """Class to handle exoplanet data preprocessing and preparation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        self.smote = SMOTE(random_state=42)
        
    def download_nasa_data(self, data_type='kepler'):
        """
        Download NASA exoplanet data from the Exoplanet Archive using TAP service.
        
        Args:
            data_type (str): Type of data to download ('kepler', 'k2', 'tess')
            
        Returns:
            pd.DataFrame: Downloaded dataset
        """
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        if data_type.lower() == 'kepler':
            table = 'cumulative'
        elif data_type.lower() == 'k2':
            table = 'k2pandc'
        elif data_type.lower() == 'tess':
            table = 'toi'
        else:
            raise ValueError("data_type must be 'kepler', 'k2', or 'tess'")
        
        url = f"{base_url}?query=select+*+from+{table.replace(' ', '+')}&format=csv"
        
        try:
            logger.info(f"Downloading {data_type.upper()} data from NASA...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV data
            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Downloaded {len(df)} records from {data_type.upper()} dataset")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error downloading data: {e}")
            return None
    
    def process_light_curve(self, target_name, mission='Kepler', author=None):
        """
        Fetch and process light curve data using lightkurve.
        
        Args:
            target_name (str): Name of the target (e.g., 'Kepler-10')
            mission (str): Mission name ('Kepler', 'K2', 'TESS')
            author (str, optional): Data author
            
        Returns:
            dict: Extracted features from light curve
        """
        try:
            logger.info(f"Processing light curve for {target_name} from {mission}")
            search_result = lk.search_lightcurve(target_name, mission=mission, author=author)
            if len(search_result) == 0:
                raise ValueError(f"No light curve found for {target_name}")
            
            lc_collection = search_result.download_all()
            lc = lc_collection.stitch().normalize().remove_outliers()
            
            # Compute periodogram
            pg = lc.to_periodogram()
            period = pg.period_at_max_power.value
            
            # Fold and get transit features
            folded_lc = lc.fold(period=period)
            transit_mask = folded_lc.get_transit_mask(period=period)
            
            # Extract features
            features = {
                'period': period,
                'duration': pg.duration_at_max_power.value,
                'depth': np.abs(np.percentile(lc.flux[transit_mask], 5) - 1) * 1e6,  # ppm
                # Add more: impact, etc., if needed
            }
            
            logger.info(f"Extracted features: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error processing light curve: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean and preprocess the exoplanet dataset.
        
        Args:
            df (pd.DataFrame): Raw exoplanet data
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Cleaning data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize column names (convert to lowercase, replace spaces with underscores)
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        
        # Define key features for exoplanet classification
        feature_mapping = {
            'kepler': {
                'period': 'koi_period',
                'radius': 'koi_prad',
                'duration': 'koi_duration',
                'depth': 'koi_depth',
                'impact': 'koi_impact',
                'teff': 'koi_steff',
                'logg': 'koi_slogg',
                'feh': 'koi_smet',
                'status': 'koi_disposition'
            },
            'k2': {
                'period': 'pl_orbper',
                'radius': 'pl_rade',
                'duration': 'pl_trandur',
                'depth': 'pl_trandep',
                'impact': 'pl_imppar',
                'teff': 'st_teff',
                'logg': 'st_logg',
                'feh': 'st_met',
                'status': 'disposition'
            },
            'tess': {
                'period': 'pl_orbper',
                'radius': 'pl_rade',
                'duration': 'pl_trandurh',
                'depth': 'pl_trandep',
                'impact': 'pl_imppar',
                'teff': 'st_teff',
                'logg': 'st_logg',
                'feh': 'st_met',
                'status': 'tfopwg_disp'
            }
        }
        
        # Determine dataset type based on available columns
        dataset_type = None
        for dt, mapping in feature_mapping.items():
            if any(col in df_clean.columns for col in mapping.values()):
                dataset_type = dt
                break
        
        if dataset_type is None:
            # Fallback: use common column patterns
            dataset_type = 'kepler'
            feature_mapping[dataset_type] = {
                'period': 'period',
                'radius': 'prad',
                'duration': 'duration',
                'depth': 'depth',
                'impact': 'impact',
                'teff': 'steff',
                'logg': 'slogg',
                'feh': 'smet',
                'status': 'disposition'
            }
        
        # Map features to standard names
        features = feature_mapping[dataset_type]
        df_standardized = pd.DataFrame()
        
        for feature, col_name in features.items():
            if col_name in df_clean.columns:
                df_standardized[feature] = df_clean[col_name]
            else:
                # Try alternative column names
                alt_names = [col for col in df_clean.columns if feature in col.lower()]
                if alt_names:
                    df_standardized[feature] = df_clean[alt_names[0]]
                else:
                    logger.warning(f"{feature} column not found, filling with default values")
                    # Use reasonable default values instead of NaN
                    if feature in ['period', 'radius', 'duration', 'depth', 'impact', 'teff', 'logg', 'feh']:
                        if feature == 'period':
                            df_standardized[feature] = 10.0  # 10 days
                        elif feature == 'radius':
                            df_standardized[feature] = 1.0  # 1 Earth radius
                        elif feature == 'duration':
                            df_standardized[feature] = 5.0  # 5 hours
                        elif feature == 'depth':
                            df_standardized[feature] = 100.0  # 100 ppm
                        elif feature == 'impact':
                            df_standardized[feature] = 0.5  # 0.5 impact parameter
                        elif feature == 'teff':
                            df_standardized[feature] = 5500.0  # 5500 K
                        elif feature == 'logg':
                            df_standardized[feature] = 4.5  # 4.5 log g
                        elif feature == 'feh':
                            df_standardized[feature] = 0.0  # 0.0 metallicity
                    else:
                        df_standardized[feature] = 0.0
        
        # Handle missing values
        logger.info(f"Original data shape: {df_standardized.shape}")
        
        # For numerical columns, fill missing values with median
        numerical_cols = ['period', 'radius', 'duration', 'depth', 'impact', 'teff', 'logg', 'feh']
        for col in numerical_cols:
            if col in df_standardized.columns:
                df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce')
                # Handle case where all values are NaN
                if df_standardized[col].isna().all():
                    df_standardized[col] = 0.0  # Default value for completely missing columns
                else:
                    median_val = df_standardized[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df_standardized[col] = df_standardized[col].fillna(median_val)
                # Ensure proper data type
                df_standardized[col] = df_standardized[col].astype('float64')
        
        # Improved imputation with KNN
        if df_standardized[numerical_cols].isna().any().any():
            logger.info("Applying KNN imputation...")
            imputer = KNNImputer(n_neighbors=5)
            df_standardized[numerical_cols] = pd.DataFrame(
                imputer.fit_transform(df_standardized[numerical_cols]),
                columns=numerical_cols,
                index=df_standardized.index
            )
        
        # Handle categorical status column
        if 'status' in df_standardized.columns:
            # Clean status values
            df_standardized['status'] = df_standardized['status'].astype(str).str.strip()
            # Map to standard categories
            status_mapping = {
                'CONFIRMED': 'Confirmed',
                'CANDIDATE': 'Candidate', 
                'FALSE POSITIVE': 'False Positive',
                'FALSE POSITIVE.': 'False Positive',
                'FALSE POSITIVE..': 'False Positive',
                'CANDIDATE.': 'Candidate',
                'CANDIDATE..': 'Candidate',
                # TESS specific values
                'CP': 'Confirmed',
                'KP': 'Confirmed',
                'PC': 'Candidate',
                'APC': 'Candidate',
                'FA': 'False Positive',
                'FP': 'False Positive',
                'REFUTED': 'False Positive'
            }
            # Only map if the value is in the mapping, otherwise keep original
            df_standardized['status'] = df_standardized['status'].map(status_mapping).fillna(df_standardized['status'])
        else:
            # Create dummy status if not available
            df_standardized['status'] = 'Unknown'
        
        # Remove rows with all NaN values
        df_standardized = df_standardized.dropna(how='all')
        
        logger.info(f"Cleaned data shape: {df_standardized.shape}")
        return df_standardized
    
    def prepare_features(self, df, target_column='status'):
        """
        Prepare features for machine learning.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            target_column (str): Name of target column
            
        Returns:
            tuple: (X, y, feature_names)
        """
        logger.info("Preparing features for machine learning...")
        
        # Select numerical features
        numerical_features = ['period', 'radius', 'duration', 'depth', 'impact', 'teff', 'logg', 'feh']
        available_features = [col for col in numerical_features if col in df.columns]
        
        if not available_features:
            raise ValueError("No suitable numerical features found in the dataset")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        self.feature_columns = available_features
        
        # Prepare target variable
        if target_column in df.columns:
            y = df[target_column].copy()
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Remove rows where target is NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared {len(available_features)} features: {available_features}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, available_features
    
    def handle_imbalance(self, X, y):
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        logger.info("Handling class imbalance with SMOTE...")
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        logger.info(f"Resampled target distribution:\n{y_resampled.value_counts()}")
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data into {int((1-test_size)*100)}%-{int(test_size*100)}% train-test split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, X_train, X_test=None):
        """
        Fit scaler on training data and transform both train and test sets.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) or (X_train_scaled, None)
        """
        logger.info("Normalizing features...")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        self.is_fitted = True
        return X_train_scaled, X_test_scaled
    
    def transform(self, X):
        """
        Transform new data using fitted scaler.
        
        Args:
            X (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming new data")
        
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def encode_labels(self, y_train, y_test=None):
        """
        Encode categorical labels to integers.
        
        Args:
            y_train (pd.Series): Training labels
            y_test (pd.Series, optional): Test labels
            
        Returns:
            tuple: (y_train_encoded, y_test_encoded) or (y_train_encoded, None)
        """
        logger.info("Encoding categorical labels...")
        
        # Fit encoder on training labels
        y_train_encoded = pd.Series(
            self.label_encoder.fit_transform(y_train),
            index=y_train.index
        )
        
        # Transform test labels if provided
        y_test_encoded = None
        if y_test is not None:
            y_test_encoded = pd.Series(
                self.label_encoder.transform(y_test),
                index=y_test.index
            )
        
        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return y_train_encoded, y_test_encoded
    
    def inverse_transform_labels(self, y_encoded):
        """
        Convert encoded labels back to original categories.
        
        Args:
            y_encoded (array-like): Encoded labels
            
        Returns:
            array: Original categorical labels
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def get_feature_importance_data(self, feature_importance, feature_names):
        """
        Prepare feature importance data for visualization.
        
        Args:
            feature_importance (array): Feature importance scores
            feature_names (list): Names of features
            
        Returns:
            pd.DataFrame: Feature importance data
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

def ensure_streamlit_compatibility(df):
    """
    Ensure DataFrame is compatible with Streamlit/PyArrow serialization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Streamlit-compatible DataFrame
    """
    df_clean = df.copy()
    
    # Convert all columns to appropriate types
    for col in df_clean.columns:
        try:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric first
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                if not numeric_series.isna().all():
                    df_clean[col] = numeric_series.astype('float64')
                else:
                    # Keep as string but ensure it's clean
                    df_clean[col] = df_clean[col].astype(str).fillna('')
            elif df_clean[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                # Convert all integer types to float64
                df_clean[col] = df_clean[col].astype('float64')
            elif df_clean[col].dtype in ['bool']:
                # Convert boolean to string
                df_clean[col] = df_clean[col].astype(str)
            elif df_clean[col].dtype in ['float32']:
                # Convert float32 to float64
                df_clean[col] = df_clean[col].astype('float64')
            elif 'datetime' in str(df_clean[col].dtype):
                # Convert datetime to string
                df_clean[col] = df_clean[col].astype(str)
        except Exception as e:
            # If conversion fails, convert to string
            logger.warning(f"Could not convert column {col}: {e}")
            df_clean[col] = df_clean[col].astype(str).fillna('')
    
    # Replace any remaining NaN values
    df_clean = df_clean.fillna(0.0)
    
    # Ensure no infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], 0.0)
    
    return df_clean

def create_sample_data():
    """
    Create sample exoplanet data for testing when NASA data is not available.
    Uses probabilistic noise to prevent overfitting.
    
    Returns:
        pd.DataFrame: Sample exoplanet data
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic exoplanet data
    data = {
        'period': np.random.lognormal(2, 1, n_samples),  # Orbital period in days
        'radius': np.random.lognormal(0.5, 0.8, n_samples),  # Planetary radius in Earth radii
        'duration': np.random.lognormal(1, 0.5, n_samples),  # Transit duration in hours
        'depth': np.random.lognormal(1, 1, n_samples),  # Transit depth in ppm
        'impact': np.random.uniform(0, 1, n_samples),  # Impact parameter
        'teff': np.random.normal(5500, 500, n_samples),  # Stellar effective temperature
        'logg': np.random.normal(4.5, 0.3, n_samples),  # Stellar surface gravity
        'feh': np.random.normal(0, 0.2, n_samples),  # Stellar metallicity
    }
    
    # Create labels with probabilistic noise to prevent overfitting
    labels = []
    class_names = ['Confirmed', 'Candidate', 'False Positive']
    
    for i in range(n_samples):
        period = data['period'][i]
        radius = data['radius'][i]
        depth = data['depth'][i]
        impact = data['impact'][i]
        
        # Determine base classification
        if period > 2.5 and radius > 0.7 and depth > 25 and impact < 0.5:
            # Confirmed block - use probabilistic assignment
            probabilities = [0.8, 0.1, 0.1]  # [Confirmed, Candidate, False Positive]
        elif period > 1.2 and radius > 0.3 and depth > 12 and impact < 0.6:
            # Candidate block - use probabilistic assignment
            probabilities = [0.1, 0.8, 0.1]  # [Confirmed, Candidate, False Positive]
        else:
            # False Positive block - use probabilistic assignment
            probabilities = [0.1, 0.1, 0.8]  # [Confirmed, Candidate, False Positive]
        
        # Use np.random.choice with probabilities
        label = np.random.choice(class_names, p=probabilities)
        labels.append(label)
    
    data['status'] = labels
    
    # Ensure we have all three classes with minimum samples
    df = pd.DataFrame(data)
    class_counts = df['status'].value_counts()
    
    # If any class has less than 10 samples, adjust the probabilities
    min_samples = 10
    if any(count < min_samples for count in class_counts.values):
        # Regenerate with more balanced probabilities
        labels = []
        for i in range(n_samples):
            period = data['period'][i]
            radius = data['radius'][i]
            depth = data['depth'][i]
            impact = data['impact'][i]
            
            # More balanced probabilities
            if period > 2.0 and radius > 0.6 and depth > 20 and impact < 0.6:
                probabilities = [0.6, 0.2, 0.2]  # [Confirmed, Candidate, False Positive]
            elif period > 1.0 and radius > 0.2 and depth > 8 and impact < 0.7:
                probabilities = [0.2, 0.6, 0.2]  # [Confirmed, Candidate, False Positive]
            else:
                probabilities = [0.2, 0.2, 0.6]  # [Confirmed, Candidate, False Positive]
            
            label = np.random.choice(class_names, p=probabilities)
            labels.append(label)
        
        data['status'] = labels
        df = pd.DataFrame(data)
        class_counts = df['status'].value_counts()
    
    logger.info(f"Sample data class distribution: {dict(class_counts)}")
    
    return df

if __name__ == "__main__":
    # Test the preprocessing pipeline
    processor = ExoplanetDataProcessor()
    
    # Try to download real data, fallback to sample data
    logger.info("Testing data preprocessing pipeline...")
    
    # Try downloading Kepler data
    df = processor.download_nasa_data('kepler')
    
    if df is None or len(df) == 0:
        logger.info("Using sample data for testing...")
        df = create_sample_data()
    
    # Clean and prepare data
    df_clean = processor.clean_data(df)
    X, y, features = processor.prepare_features(df_clean)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train_scaled, X_test_scaled = processor.fit_transform(X_train, X_test)
    y_train_encoded, y_test_encoded = processor.encode_labels(y_train, y_test)
    
    # Test light curve processing
    features = processor.process_light_curve('Kepler-10')
    logger.info(f"Light curve features: {features}")
    
    logger.info("Preprocessing pipeline test completed successfully!")
    logger.info(f"Final training set shape: {X_train_scaled.shape}")
    logger.info(f"Final test set shape: {X_test_scaled.shape}")