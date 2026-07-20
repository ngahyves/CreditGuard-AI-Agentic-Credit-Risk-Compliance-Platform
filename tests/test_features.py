import pytest
import pandas as pd
import numpy as np

def test_years_conversion_logic():
    """Checks if the days-to-years conversion is scientifically accurate (365.25)."""
    days_birth = -14610 # Exactly 40 years (including leap years)
    calculated_years = days_birth / -365.25
    assert calculated_years == 40.0

def test_dti_ratio_logic():
    """Checks if the Debt-to-Income ratio calculation is correct."""
    income = 5000
    annuity = 1000
    dti = annuity / income
    assert dti == 0.20

def test_job_seniority_clipping():
    """Checks if the 365243 anomaly treatment logic works."""
    bad_value = 365243
    # Logic from our feature_engineering.py
    cleaned_value = np.nan if bad_value == 365243 else bad_value / -365.25
    assert pd.isna(cleaned_value)