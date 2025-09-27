"""
Machine learning model for exoplanet detection.
Implements Random Forest and Gradient Boosting classifiers with evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ExoplanetClassifier:
    """Machine learning classifier for exoplanet detection."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the exoplanet classifier.
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.class_names = []
        self.training_history = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
    
    def train(self, X_train, y_train, X_test=None, y_test=None, 
              hyperparameter_tuning=False, cv_folds=5):
        """
        Train the exoplanet classification model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_test (pd.DataFrame, optional): Test features
            y_test (pd.Series, optional): Test labels
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training {self.model_type} model...")
        
        # Store feature columns and class names
        self.feature_columns = list(X_train.columns)
        self.class_names = sorted(y_train.unique())
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            self._tune_hyperparameters(X_train, y_train, cv_folds)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_predictions = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_predictions, "Training")
        
        # Evaluate on test set if provided
        test_metrics = {}
        if X_test is not None and y_test is not None:
            test_predictions = self.model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, test_predictions, "Test")
        
        # Store training history
        self.training_history = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_columns': self.feature_columns,
            'class_names': self.class_names,
            'model_type': self.model_type
        }
        
        print("Training completed successfully!")
        return self.training_history
    
    def _tune_hyperparameters(self, X_train, y_train, cv_folds=5):
        """Perform hyperparameter tuning using GridSearchCV."""
        print("Performing hyperparameter tuning...")
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        
        # Use a subset of data for faster tuning
        if len(X_train) > 5000:
            sample_indices = np.random.choice(len(X_train), 5000, replace=False)
            X_tune = X_train.iloc[sample_indices]
            y_tune = y_train.iloc[sample_indices]
        else:
            X_tune = X_train
            y_tune = y_train
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv_folds, 
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_tune, y_tune)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"\n{dataset_name} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            array: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities for new data.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature importance data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_scores = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Generate confusion matrix data.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            
        Returns:
            pd.DataFrame: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        return cm_df
    
    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation evaluation.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='f1_weighted')
        
        cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        print(f"Cross-validation F1-score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score'] * 2:.4f})")
        
        return cv_results
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'class_names': self.class_names,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.class_names = model_data['class_names']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def retrain(self, X_new, y_new, X_existing=None, y_existing=None):
        """
        Retrain the model with new data.
        
        Args:
            X_new (pd.DataFrame): New training features
            y_new (pd.Series): New training labels
            X_existing (pd.DataFrame, optional): Existing training features
            y_existing (pd.Series, optional): Existing training labels
        """
        print("Retraining model with new data...")
        
        # Combine new and existing data if provided
        if X_existing is not None and y_existing is not None:
            X_combined = pd.concat([X_existing, X_new], ignore_index=True)
            y_combined = pd.concat([y_existing, y_new], ignore_index=True)
        else:
            X_combined = X_new
            y_combined = y_new
        
        # Update feature columns and class names
        self.feature_columns = list(X_combined.columns)
        self.class_names = sorted(y_combined.unique())
        
        # Retrain the model
        self.model.fit(X_combined, y_combined)
        self.is_trained = True
        
        print("Model retraining completed!")
    
    def get_model_summary(self):
        """
        Get a summary of the trained model.
        
        Returns:
            dict: Model summary information
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        summary = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_columns),
            "class_count": len(self.class_names),
            "classes": self.class_names,
            "features": self.feature_columns
        }
        
        if self.training_history:
            summary.update({
                "training_metrics": self.training_history.get('train_metrics', {}),
                "test_metrics": self.training_history.get('test_metrics', {})
            })
        
        return summary

def create_visualization_plots(model, X_test, y_test, save_path=None):
    """
    Create visualization plots for model evaluation.
    
    Args:
        model (ExoplanetClassifier): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        save_path (str, optional): Path to save plots
    """
    if not model.is_trained:
        print("Model must be trained before creating visualizations")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exoplanet Classification Model Evaluation', fontsize=16)
    
    # 1. Confusion Matrix
    cm = model.get_confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance
    importance_df = model.get_feature_importance()
    importance_df.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Importance')
    axes[0, 1].set_xlabel('Importance Score')
    
    # 3. Prediction Confidence Distribution
    max_proba = np.max(y_pred_proba, axis=1)
    axes[1, 0].hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Prediction Confidence Distribution')
    axes[1, 0].set_xlabel('Maximum Probability')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Class Distribution
    class_counts = pd.Series(y_test).value_counts()
    axes[1, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Class Distribution in Test Set')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test the model
    from preprocess import ExoplanetDataProcessor, create_sample_data
    
    print("Testing Exoplanet Classifier...")
    
    # Create sample data
    df = create_sample_data()
    
    # Preprocess data
    processor = ExoplanetDataProcessor()
    df_clean = processor.clean_data(df)
    X, y, features = processor.prepare_features(df_clean)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    X_train_scaled, X_test_scaled = processor.fit_transform(X_train, X_test)
    y_train_encoded, y_test_encoded = processor.encode_labels(y_train, y_test)
    
    # Train model
    classifier = ExoplanetClassifier('random_forest')
    results = classifier.train(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded)
    
    # Test predictions
    predictions = classifier.predict(X_test_scaled)
    print(f"\nSample predictions: {predictions[:10]}")
    
    # Get feature importance
    importance = classifier.get_feature_importance()
    print(f"\nFeature importance:\n{importance}")
    
    print("Model testing completed successfully!")
