#!/usr/bin/env python3
"""
Preprocessing pipeline with SMOTE, RobustScaler, and stratified cross-validation
Implements the preprocessing methodology from the IRP research
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """Complete preprocessing pipeline for DDoS detection datasets"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessing pipeline
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
        self.feature_names = None
        self.is_fitted = False
        
    def clean_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Clean data: handle missing values and convert to numeric
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            
        Returns:
            Tuple of (cleaned_X, cleaned_y)
        """
        X_cleaned = X.copy()
        
        # Convert all columns to numeric where possible
        for col in X_cleaned.columns:
            X_cleaned[col] = pd.to_numeric(X_cleaned[col], errors='coerce')
        
        # Drop columns that are all NaN
        X_cleaned = X_cleaned.dropna(axis=1, how='all')
        
        # Handle remaining NaN values with median imputation
        X_cleaned = pd.DataFrame(
            self.imputer.fit_transform(X_cleaned),
            columns=X_cleaned.columns,
            index=X_cleaned.index
        )
        
        self.feature_names = X_cleaned.columns.tolist()
        
        if y is not None:
            # Convert y to numeric binary if needed
            y_cleaned = y.copy()
            if y_cleaned.dtype == 'object':
                unique_vals = y_cleaned.unique()
                y_cleaned = (y_cleaned != unique_vals[0]).astype(int)
            y_cleaned = pd.to_numeric(y_cleaned, errors='coerce').fillna(0).astype(int)
            return X_cleaned, y_cleaned
        
        return X_cleaned, None
    
    def scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale features using RobustScaler (median and IQR based)
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming test data")
            X_scaled = self.scaler.transform(X)
        
        # Verify normalization after scaling
        self._verify_scaled_features(X_scaled, fit=fit)
        
        return X_scaled
    
    def _verify_scaled_features(self, X_scaled: np.ndarray, fit: bool = False) -> Dict:
        """
        Verify that features are properly normalized after scaling
        Checks min/max values per feature
        
        Args:
            X_scaled: Scaled feature array
            fit: Whether this is the fit step (to log stats)
            
        Returns:
            Dictionary with verification results
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if len(X_scaled) == 0:
            return {}
        
        # Calculate per-feature statistics
        min_per_feature = X_scaled.min(axis=0)
        max_per_feature = X_scaled.max(axis=0)
        mean_per_feature = X_scaled.mean(axis=0)
        std_per_feature = X_scaled.std(axis=0)
        
        # For RobustScaler: values are centered around 0 with IQR-based scaling
        # Typically ranges from [-3, 3] but can be wider for outliers
        # For MinMaxScaler: values should be in [0, 1]
        
        # Check if using RobustScaler (current implementation)
        all_in_robust_range = np.all((min_per_feature >= -10.0) & (max_per_feature <= 10.0))
        all_in_01_range = np.all((min_per_feature >= -0.1) & (max_per_feature <= 1.1))
        
        if fit:  # Only log during fit to avoid spam
            n_features = X_scaled.shape[1]
            logger.debug(f"[SCALING VERIFICATION] {n_features} features scaled with RobustScaler")
            logger.debug(f"  Min values range: [{min_per_feature.min():.3f}, {min_per_feature.max():.3f}]")
            logger.debug(f"  Max values range: [{max_per_feature.min():.3f}, {max_per_feature.max():.3f}]")
            logger.debug(f"  Mean values range: [{mean_per_feature.min():.3f}, {mean_per_feature.max():.3f}]")
            logger.debug(f"  Std values range: [{std_per_feature.min():.3f}, {std_per_feature.max():.3f}]")
            
            if all_in_robust_range:
                logger.debug(f"  ✓ All features in RobustScaler expected range [-10, 10]")
            elif all_in_01_range:
                logger.warning(f"  ⚠ Features appear to be MinMaxScaled [0, 1], but using RobustScaler")
            else:
                logger.warning(f"  ⚠ Some features outside typical RobustScaler range")
        
        return {
            'min_per_feature': min_per_feature,
            'max_per_feature': max_per_feature,
            'mean_per_feature': mean_per_feature,
            'std_per_feature': std_per_feature,
            'all_in_robust_range': all_in_robust_range,
            'all_in_01_range': all_in_01_range
        }
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance classes
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    apply_smote_flag: bool = True,
                    scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            X: Feature dataframe
            y: Target series
            apply_smote_flag: Whether to apply SMOTE
            scale: Whether to scale features
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        # Clean data
        X_cleaned, y_cleaned = self.clean_data(X, y)
        
        # Convert to numpy
        X_array = X_cleaned.values
        y_array = y_cleaned.values
        
        # Apply SMOTE if requested
        if apply_smote_flag:
            X_array, y_array = self.apply_smote(X_array, y_array)
        
        # Scale features
        if scale:
            X_array = self.scale_features(X_array, fit=True)
        
        return X_array, y_array
    
    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessing steps
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data")
        
        # Clean (using fitted imputer)
        X_cleaned = X.copy()
        for col in X_cleaned.columns:
            X_cleaned[col] = pd.to_numeric(X_cleaned[col], errors='coerce')
        
        X_cleaned = pd.DataFrame(
            self.imputer.transform(X_cleaned),
            columns=X_cleaned.columns,
            index=X_cleaned.index
        )
        
        # Scale
        X_scaled = self.scale_features(X_cleaned.values, fit=False)
        
        return X_scaled


class StratifiedCrossValidator:
    """Stratified 5-fold cross-validation for model evaluation"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize stratified cross-validator
        
        Args:
            n_splits: Number of folds (default: 5)
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Generate train/test splits
        
        Args:
            X: Feature array
            y: Target array
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        return self.skf.split(X, y)
    
    def get_folds(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Get all fold splits as a list
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            List of dictionaries with 'train' and 'test' indices
        """
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X, y)):
            folds.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'test_idx': test_idx
            })
        return folds


def main():
    """Test the preprocessing pipeline"""
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader()
    
    try:
        df = loader.load_ton_iot()
        print(f"Dataset loaded: {df.shape}\n")
        
        # Prepare data
        X = df.drop(['label', 'type'] if 'type' in df.columns else ['label'], axis=1, errors='ignore')
        y = df['label']
        
        # Initialize pipeline
        pipeline = PreprocessingPipeline(random_state=42)
        
        # Preprocess
        X_processed, y_processed = pipeline.prepare_data(X, y, apply_smote_flag=True, scale=True)
        
        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        print(f"Processed class distribution: {pd.Series(y_processed).value_counts().to_dict()}\n")
        
        # Test cross-validation
        cv = StratifiedCrossValidator(n_splits=5, random_state=42)
        folds = cv.get_folds(X_processed, y_processed)
        
        print(f"5-fold CV splits generated: {len(folds)} folds")
        for fold in folds:
            print(f"  Fold {fold['fold']}: Train={len(fold['train_idx'])}, Test={len(fold['test_idx'])}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
