import pytest
import pandas as pd
import pandera as pa

def test_pydantic_contract_logic():
    """Ensures our expected data types are strictly enforced."""
    schema = pa.DataFrameSchema({
        "SK_ID_CURR": pa.Column(int, unique=True),
        "TARGET": pa.Column(int, pa.Check.isin([0, 1])),
        "AMT_INCOME_TOTAL": pa.Column(float, pa.Check.ge(0))
    })
    
    # 1. Valid Data
    valid_df = pd.DataFrame({
        "SK_ID_CURR": [100002],
        "TARGET": [1],
        "AMT_INCOME_TOTAL": [50000.0]
    })
    assert schema.validate(valid_df) is not None
    
    # 2. Invalid Data (Negative Income)
    with pytest.raises(pa.errors.SchemaError):
        invalid_df = pd.DataFrame({
            "SK_ID_CURR": [100002],
            "TARGET": [1],
            "AMT_INCOME_TOTAL": [-100.0]
        })
        schema.validate(invalid_df)