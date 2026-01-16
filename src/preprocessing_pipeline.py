#!/usr/bin/env python3
"""
Preprocessing pipeline with complete workflow:
1. Data Cleaning (NaN, Infinity removal)
2. Encoding (categorical features)
3. Feature Selection (mutual information, correlation)
4. Scaling (RobustScaler)
5. Resampling (SMOTE for class balancing)
6. Splitting (Train/Validation/Test with stratification)

Implements the preprocessing methodology from the IRP research
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional, Dict
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for DDoS detection datasets
    
    Workflow:
    1. Data Cleaning: Remove NaN, Infinity, convert to numeric
    2. Encoding: Encode categorical features (if any)
    3. Feature Selection: Select top K features using mutual information
    4. Scaling: Normalize features with RobustScaler
    5. Resampling: Balance classes with SMOTE
    6. Splitting: Split into Train/Validation/Test (stratified)
    """
    
    def __init__(self, random_state: int = 42, n_features: int = 20):
        """
        Initialize preprocessing pipeline
        
        Args:
            random_state: Random seed for reproducibility
            n_features: Number of features to select (default: 20)
        """
        self.random_state = random_state
        self.n_features = n_features
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
        self.feature_selector = None
        self.label_encoders = {}
        self.feature_names = None
        self.selected_features = None
        self.is_fitted = False
        
    def clean_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Step 1: Data Cleaning
        - Remove NaN and Infinity values
        - Convert to numeric
        - Drop columns that are all NaN
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            
        Returns:
            Tuple of (cleaned_X, cleaned_y)
        """
        logger.info("[PREPROCESSING] Step 1: Data Cleaning...")
        X_cleaned = X.copy()
        
        # Convert all columns to numeric where possible
        for col in X_cleaned.columns:
            X_cleaned[col] = pd.to_numeric(X_cleaned[col], errors='coerce')
        
        # Remove Infinity values (replace with NaN)
        X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns that are all NaN
        cols_before = len(X_cleaned.columns)
        X_cleaned = X_cleaned.dropna(axis=1, how='all')
        cols_after = len(X_cleaned.columns)
        if cols_before != cols_after:
            logger.info(f"  Dropped {cols_before - cols_after} columns (all NaN)")
        
        # Handle remaining NaN values with median imputation
        X_cleaned = pd.DataFrame(
            self.imputer.fit_transform(X_cleaned),
            columns=X_cleaned.columns,
            index=X_cleaned.index
        )
        
        self.feature_names = X_cleaned.columns.tolist()
        logger.info(f"  Cleaned data: {X_cleaned.shape[0]} rows, {X_cleaned.shape[1]} features")
        
        if y is not None:
            # Ensure y is binary (0/1)
            y_cleaned = y.copy()
            if y_cleaned.dtype == 'object':
                y_cleaned = pd.to_numeric(y_cleaned, errors='coerce').fillna(0)
            y_cleaned = y_cleaned.astype(int)
            logger.info(f"  Label distribution: {y_cleaned.value_counts().to_dict()}")
            return X_cleaned, y_cleaned
        
        return X_cleaned, None
    
    def encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Encoding
        - Encode categorical features using LabelEncoder
        
        Args:
            X: Feature dataframe
            
        Returns:
            Encoded dataframe
        """
        logger.info("[PREPROCESSING] Step 2: Encoding categorical features...")
        X_encoded = X.copy()
        
        # Identify categorical columns (non-numeric or object type)
        categorical_cols = []
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_encoded[col]):
                categorical_cols.append(col)
        
        if categorical_cols:
            logger.info(f"  Found {len(categorical_cols)} categorical features: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                # Handle NaN values before encoding
                X_encoded[col] = X_encoded[col].fillna('unknown')
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
        else:
            logger.info("  No categorical features found (all numeric)")
        
        return X_encoded
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """
        Step 3: Feature Selection
        - Select top K features using mutual information
        
        Args:
            X: Feature dataframe
            y: Target series
            fit: Whether to fit the selector (True for training data)
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"[PREPROCESSING] Step 3: Feature Selection (selecting top {self.n_features} features)...")
        
        if fit:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X.values, y.values, random_state=self.random_state)
            
            # Select top K features
            self.feature_selector = SelectKBest(k=min(self.n_features, len(X.columns)))
            X_selected = self.feature_selector.fit_transform(X.values, y.values)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]
            
            logger.info(f"  Selected {len(self.selected_features)} features from {len(X.columns)}")
            logger.info(f"  Top features: {self.selected_features[:10]}{'...' if len(self.selected_features) > 10 else ''}")
            
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector must be fitted before transforming test data")
            X_selected = self.feature_selector.transform(X.values)
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Step 4: Scaling
        - Scale features using RobustScaler (median and IQR based)
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled feature array
        """
        logger.info("[PREPROCESSING] Step 4: Scaling features with RobustScaler...")
        
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
        """Verify that features are properly normalized after scaling"""
        if len(X_scaled) == 0:
            return {}
        
        min_per_feature = X_scaled.min(axis=0)
        max_per_feature = X_scaled.max(axis=0)
        mean_per_feature = X_scaled.mean(axis=0)
        std_per_feature = X_scaled.std(axis=0)
        
        all_in_robust_range = np.all((min_per_feature >= -10.0) & (max_per_feature <= 10.0))
        
        if fit:
            logger.debug(f"  Min range: [{min_per_feature.min():.3f}, {min_per_feature.max():.3f}]")
            logger.debug(f"  Max range: [{max_per_feature.min():.3f}, {max_per_feature.max():.3f}]")
            logger.debug(f"  Mean range: [{mean_per_feature.min():.3f}, {mean_per_feature.max():.3f}]")
            if all_in_robust_range:
                logger.debug("  ✓ All features in RobustScaler expected range [-10, 10]")
        
        return {
            'min_per_feature': min_per_feature,
            'max_per_feature': max_per_feature,
            'mean_per_feature': mean_per_feature,
            'std_per_feature': std_per_feature,
            'all_in_robust_range': all_in_robust_range
        }
    
    def resample_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 5: Resampling
        - Balance classes using SMOTE
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info("[PREPROCESSING] Step 5: Resampling with SMOTE...")
        logger.info(f"  Before resampling: {pd.Series(y).value_counts().to_dict()}")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        logger.info(f"  After resampling: {pd.Series(y_resampled).value_counts().to_dict()}")
        logger.info(f"  Shape: {X_resampled.shape[0]} rows, {X_resampled.shape[1]} features")
        
        return X_resampled, y_resampled
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Step 6: Splitting
        - Split data into Train/Validation/Test sets with stratification
        
        Args:
            X: Feature array
            y: Target array
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing (X, y) tuple
        """
        logger.info("[PREPROCESSING] Step 6: Splitting data (Train/Validation/Test)...")
        
        # Validate ratios sum to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # First split: train + (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(val_ratio + test_ratio),
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            stratify=y_temp,
            random_state=self.random_state
        )
        
        logger.info(f"  Training set: {X_train.shape[0]} samples (class distribution: {pd.Series(y_train).value_counts().to_dict()})")
        logger.info(f"  Validation set: {X_val.shape[0]} samples (class distribution: {pd.Series(y_val).value_counts().to_dict()})")
        logger.info(f"  Test set: {X_test.shape[0]} samples (class distribution: {pd.Series(y_test).value_counts().to_dict()})")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    apply_encoding: bool = True,
                    apply_feature_selection: bool = True,
                    apply_scaling: bool = True,
                    apply_resampling: bool = True,
                    apply_splitting: bool = True,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            X: Feature dataframe
            y: Target series
            apply_encoding: Whether to encode categorical features
            apply_feature_selection: Whether to select features
            apply_scaling: Whether to scale features
            apply_resampling: Whether to resample (SMOTE)
            apply_splitting: Whether to split into train/val/test
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Data Cleaning
        X_cleaned, y_cleaned = self.clean_data(X, y)
        
        # Step 2: Encoding
        if apply_encoding:
            X_encoded = self.encode_features(X_cleaned)
        else:
            X_encoded = X_cleaned
        
        # Step 3: Feature Selection
        if apply_feature_selection:
            X_selected = self.select_features(X_encoded, y_cleaned, fit=True)
        else:
            X_selected = X_encoded
            self.selected_features = list(X_selected.columns)
        
        # Convert to numpy for scaling
        X_array = X_selected.values
        y_array = y_cleaned.values
        
        # Step 4: Scaling
        if apply_scaling:
            X_scaled = self.scale_features(X_array, fit=True)
        else:
            X_scaled = X_array
        
        # Step 5: Resampling
        if apply_resampling:
            X_resampled, y_resampled = self.resample_data(X_scaled, y_array)
        else:
            X_resampled, y_resampled = X_scaled, y_array
        
        # Step 6: Splitting
        result = {
            'X_processed': X_resampled,
            'y_processed': y_resampled,
            'feature_names': self.selected_features,
            'preprocessing_steps': {
                'cleaning': True,
                'encoding': apply_encoding,
                'feature_selection': apply_feature_selection,
                'scaling': apply_scaling,
                'resampling': apply_resampling,
                'splitting': False
            }
        }
        
        if apply_splitting:
            splits = self.split_data(X_resampled, y_resampled, train_ratio, val_ratio, test_ratio)
            result['splits'] = splits
            result['preprocessing_steps']['splitting'] = True
        else:
            result['splits'] = None
        
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return result
    
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
        
        # Clean
        X_cleaned = X.copy()
        for col in X_cleaned.columns:
            X_cleaned[col] = pd.to_numeric(X_cleaned[col], errors='coerce')
        X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
        X_cleaned = pd.DataFrame(
            self.imputer.transform(X_cleaned),
            columns=X_cleaned.columns,
            index=X_cleaned.index
        )
        
        # Encode (if needed)
        for col, encoder in self.label_encoders.items():
            if col in X_cleaned.columns:
                X_cleaned[col] = X_cleaned[col].fillna('unknown')
                X_cleaned[col] = encoder.transform(X_cleaned[col].astype(str))
        
        # Select features
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_cleaned.values)
        else:
            X_selected = X_cleaned.values
        
        # Scale
        X_scaled = self.scale_features(X_selected, fit=False)
        
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
        pipeline = PreprocessingPipeline(random_state=42, n_features=20)
        
        # Preprocess
        result = pipeline.prepare_data(
            X, y,
            apply_encoding=True,
            apply_feature_selection=True,
            apply_scaling=True,
            apply_resampling=True,
            apply_splitting=True
        )
        
        print(f"\nOriginal shape: {X.shape}")
        print(f"Processed shape: {result['X_processed'].shape}")
        print(f"Selected features: {len(result['feature_names'])}")
        
        if result['splits']:
            X_train, y_train = result['splits']['train']
            X_val, y_val = result['splits']['val']
            X_test, y_test = result['splits']['test']
            print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
