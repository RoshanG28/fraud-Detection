"""
Fraud Detection Machine Learning Model
Author: Cyril Anand
Description: Ensemble ML model for transaction fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Ensemble fraud detection model
    
    Features:
    - Multiple ML algorithms (Logistic Regression, Random Forest)
    - Handles class imbalance with SMOTE
    - Feature engineering and selection
    - Model evaluation and metrics
    """
    
    def __init__(self):
        """Initialize fraud detection model"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model = None
        self.is_trained = False
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for fraud detection
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Amount-based features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
            
            # Rolling statistics (if timestamp available)
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
                df['rolling_avg_7d'] = df.groupby('card_id')['amount'].transform(
                    lambda x: x.rolling(7, min_periods=1).mean()
                )
                df['rolling_std_7d'] = df.groupby('card_id')['amount'].transform(
                    lambda x: x.rolling(7, min_periods=1).std()
                )
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Transaction frequency
        if 'card_id' in df.columns and 'timestamp' in df.columns:
            df['days_since_last'] = df.groupby('card_id')['timestamp'].diff().dt.days
            df['transaction_count_24h'] = df.groupby('card_id').rolling(
                '24H', on='timestamp'
            ).size().reset_index(drop=True)
        
        # Distance-based features (if location available)
        if 'merchant_lat' in df.columns and 'card_lat' in df.columns:
            df['distance_km'] = self._haversine_distance(
                df['card_lat'], df['card_lon'],
                df['merchant_lat'], df['merchant_lon']
            )
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['timestamp', 'card_id', 'merchant_id']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info(f"Feature engineering complete. Features: {df.shape[1]}")
        return df
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str = 'is_fraud',
                    test_size: float = 0.2,
                    balance: bool = True) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Test set proportion
            balance: Whether to balance classes with SMOTE
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data...")
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features
        exclude_cols = [target_col, 'timestamp', 'card_id', 'merchant_id']
        exclude_cols.extend([col for col in df.columns if col.endswith('_id')])
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Handle class imbalance
        if balance:
            logger.info("Balancing classes with SMOTE...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: Fraud={sum(y_train)}, Legit={len(y_train)-sum(y_train)}")
        
        logger.info("Data preparation complete")
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, optimize: bool = False):
        """
        Train ensemble fraud detection model
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize: Whether to perform hyperparameter optimization
        """
        logger.info("Training fraud detection model...")
        
        # Define base models
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Create ensemble
        self.model = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('rf', rf)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # Hyperparameter optimization
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            param_grid = {
                'lr__C': [0.1, 1.0, 10.0],
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [10, 20, 30]
            }
            
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("Model training complete")
    
    def evaluate(self, X_test, y_test) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # Calculate false positive rate reduction
        fpr = metrics['confusion_matrix']['false_positive'] / (
            metrics['confusion_matrix']['false_positive'] + 
            metrics['confusion_matrix']['true_negative']
        )
        metrics['false_positive_rate'] = fpr
        
        # Feature importance (for Random Forest)
        if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
            importances = self.model.named_estimators_['rf'].feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            metrics['top_features'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        
        logger.info("Evaluation complete")
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall: {metrics['recall']:.3f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.3f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def predict(self, X, return_proba: bool = False):
        """
        Make predictions on new data
        
        Args:
            X: Features DataFrame
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Engineer features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Ensure correct features
        X = X[self.feature_names].fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict(X_scaled)
    
    def save_model(self, filepath: str = 'fraud_detection_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'fraud_detection_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


def generate_sample_data(n_samples: int = 50000) -> pd.DataFrame:
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    # Generate legitimate transactions
    n_legit = int(n_samples * 0.98)
    n_fraud = n_samples - n_legit
    
    # Legitimate transactions
    legit_data = {
        'amount': np.random.lognormal(mean=4, sigma=1, size=n_legit),
        'hour': np.random.choice(range(8, 22), size=n_legit),
        'day_of_week': np.random.choice(range(7), size=n_legit),
        'merchant_category': np.random.choice(['retail', 'food', 'gas', 'online'], size=n_legit),
        'card_present': np.random.choice([0, 1], size=n_legit, p=[0.3, 0.7]),
        'is_fraud': np.zeros(n_legit)
    }
    
    # Fraudulent transactions (different patterns)
    fraud_data = {
        'amount': np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),
        'hour': np.random.choice(range(24), size=n_fraud),
        'day_of_week': np.random.choice(range(7), size=n_fraud),
        'merchant_category': np.random.choice(['online', 'international', 'gas'], size=n_fraud),
        'card_present': np.random.choice([0, 1], size=n_fraud, p=[0.9, 0.1]),
        'is_fraud': np.ones(n_fraud)
    }
    
    # Combine
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Example usage"""
    
    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(n_samples=50000)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Initialize model
    model = FraudDetectionModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(df, balance=True)
    
    # Train model
    model.train(X_train, y_train, optimize=False)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save_model('models/fraud_detection_model.pkl')
    
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL - TRAINING COMPLETE")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Precision: {metrics['precision']:.1%}")
    print(f"Recall: {metrics['recall']:.1%}")
    print(f"F1-Score: {metrics['f1_score']:.1%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
