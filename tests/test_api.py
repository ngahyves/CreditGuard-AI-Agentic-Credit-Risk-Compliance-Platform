import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health_endpoint():
    """Verifies that the infrastructure probe is alive."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_metrics_endpoint():
    """Verifies that Prometheus metrics are exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "loan_appraisals_total" in response.text