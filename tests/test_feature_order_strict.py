import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def test_feature_order_strict_transform():
    # Test that ColumnTransformer fails if columns are missing or in wrong order (if not handled)
    # Actually ColumnTransformer handles order if passed a DataFrame with named columns.
    # But we want to ensure our feature_order is respected.
    
    features = ["feat1", "feat2"]
    ct = ColumnTransformer([("scaler", StandardScaler(), features)])
    
    df_train = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    ct.fit(df_train)
    
    # Missing column
    df_test_missing = pd.DataFrame({"feat1": [5, 6]})
    with pytest.raises(ValueError):
        ct.transform(df_test_missing)
        
    # Correct columns
    df_test_ok = pd.DataFrame({"feat1": [5, 6], "feat2": [7, 8]})
    X_trans = ct.transform(df_test_ok)
    assert X_trans.shape == (2, 2)
