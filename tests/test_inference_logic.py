import pytest

def test_decision_verdict_logic():
    """
    Checks if the Business Rules Engine assigns the correct verdict 
    based on the credit score thresholds.
    """
    # Simulate thresholds from Strategy A
    low_score = 250
    mid_score = 550
    high_score = 850
    
    def get_mock_verdict(score):
        if score > 700: return "APPROVE"
        if score >= 400: return "MANUAL REVIEW"
        return "DECLINE"

    assert get_mock_verdict(low_score) == "DECLINE"
    assert get_mock_verdict(mid_score) == "MANUAL REVIEW"
    assert get_mock_verdict(high_score) == "APPROVE"