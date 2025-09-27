"""
Data preprocessing module for exoplanet detection.
Handles data cleaning, normalization, and preparation for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import io
import zipfile
import os

class ExoplanetDataProcessor:
    """Class to handle exoplanet data preprocessing and preparation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        
    def download_nasa_data(self, data_type='kepler'):
        """
        Download NASA exoplanet data from the Exoplanet Archive.
        
        Args:
            data_type (str): Type of data to download ('kepler', 'k2', 'tess')
            
        Returns:
            pd.DataFrame: Downloaded dataset
        """
        base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        
        if data_type.lower() == 'kepler':
            url = f"{base_url}?table=cumulative&format=csv"
        elif data_type.lower() == 'k2':
            url = f"{base_url}?table=k2candidates&format=csv"
        elif data_type.lower() == 'tess':
            url = f"{base_url}?table=toi&format=csv"
        else:
            raise ValueError("data_type must be 'kepler', 'k2', or 'tess'")
        
        try:
            print(f"Downloading {data_type.upper()} data from NASA...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV data
            df = pd.read_csv(io.StringIO(response.text))
            print(f"Downloaded {len(df)} records from {data_type.upper()} dataset")
            return df
            
        except requests.RequestException as e:
            print(f"Error downloading data: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean and preprocess the exoplanet dataset.
        
        Args:
            df (pd.DataFrame): Raw exoplanet data
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Cleaning data...")
        
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
                'period': 'k2_period',
                'radius': 'k2_prad',
                'duration': 'k2_duration',
                'depth': 'k2_depth',
                'impact': 'k2_impact',
                'teff': 'k2_steff',
                'logg': 'k2_slogg',
                'feh': 'k2_smet',
                'status': 'k2_disposition'
            },
            'tess': {
                'period': 'toi_period',
                'radius': 'toi_prad',
                'duration': 'toi_duration',
                'depth': 'toi_depth',
                'impact': 'toi_impact',
                'teff': 'toi_steff',
                'logg': 'toi_slogg',
                'feh': 'toi_smet',
                'status': 'toi_disposition'
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
                    print(f"Warning: {feature} column not found, filling with NaN")
                    df_standardized[feature] = np.nan
        
        # Handle missing values
        print(f"Original data shape: {df_standardized.shape}")
        
        # For numerical columns, fill missing values with median
        numerical_cols = ['period', 'radius', 'duration', 'depth', 'impact', 'teff', 'logg', 'feh']
        for col in numerical_cols:
            if col in df_standardized.columns:
                df_standardized[col] = pd.to_numeric(df_standardized[col], errors='coerce')
                df_standardized[col] = df_standardized[col].fillna(df_standardized[col].median())
        
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
                'CANDIDATE..': 'Candidate'
            }
            # Only map if the value is in the mapping, otherwise keep original
            df_standardized['status'] = df_standardized['status'].map(status_mapping).fillna(df_standardized['status'])
        else:
            # Create dummy status if not available
            df_standardized['status'] = 'Unknown'
        
        # Remove rows with all NaN values
        df_standardized = df_standardized.dropna(how='all')
        
        print(f"Cleaned data shape: {df_standardized.shape}")
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
        print("Preparing features for machine learning...")
        
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
        
        print(f"Prepared {len(available_features)} features: {available_features}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, available_features
    
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
        print(f"Splitting data into {int((1-test_size)*100)}%-{int(test_size*100)}% train-test split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
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
        print("Normalizing features...")
        
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
        print("Encoding categorical labels...")
        
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
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
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

def create_sample_data():
    """
    Create sample exoplanet data for testing when NASA data is not available.
    
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
    
    # Create labels with more balanced distribution
    labels = []
    for i in range(n_samples):
        period = data['period'][i]
        radius = data['radius'][i]
        depth = data['depth'][i]
        impact = data['impact'][i]
        
        # More balanced classification rules
        if period > 2.5 and radius > 0.7 and depth > 25 and impact < 0.5:
            labels.append('Confirmed')
        elif period > 1.2 and radius > 0.3 and depth > 12 and impact < 0.6:
            labels.append('Candidate')
        else:
            labels.append('False Positive')
    
    data['status'] = labels
    
    # Ensure we have all three classes with minimum samples
    df = pd.DataFrame(data)
    class_counts = df['status'].value_counts()
    
    # If any class has less than 10 samples, adjust the thresholds
    min_samples = 10
    if any(count < min_samples for count in class_counts.values):
        # Regenerate with more lenient thresholds
        labels = []
        for i in range(n_samples):
            period = data['period'][i]
            radius = data['radius'][i]
            depth = data['depth'][i]
            impact = data['impact'][i]
            
            # More lenient thresholds
            if period > 2.0 and radius > 0.6 and depth > 20 and impact < 0.6:
                labels.append('Confirmed')
            elif period > 1.0 and radius > 0.2 and depth > 8 and impact < 0.7:
                labels.append('Candidate')
            else:
                labels.append('False Positive')
        
        data['status'] = labels
        df = pd.DataFrame(data)
        class_counts = df['status'].value_counts()
    
    print(f"Sample data class distribution: {dict(class_counts)}")
    
    return df

if __name__ == "__main__":
    # Test the preprocessing pipeline
    processor = ExoplanetDataProcessor()
    
    # Try to download real data, fallback to sample data
    print("Testing data preprocessing pipeline...")
    
    # Try downloading Kepler data
    df = processor.download_nasa_data('kepler')
    
    if df is None or len(df) == 0:
        print("Using sample data for testing...")
        df = create_sample_data()
    
    # Clean and prepare data
    df_clean = processor.clean_data(df)
    X, y, features = processor.prepare_features(df_clean)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train_scaled, X_test_scaled = processor.fit_transform(X_train, X_test)
    y_train_encoded, y_test_encoded = processor.encode_labels(y_train, y_test)
    
    print("Preprocessing pipeline test completed successfully!")
    print(f"Final training set shape: {X_train_scaled.shape}")
    print(f"Final test set shape: {X_test_scaled.shape}")
