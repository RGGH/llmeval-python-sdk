"""Tests for the EvalClient."""

import pytest
from unittest.mock import Mock, patch
from llmeval import EvalClient
from llmeval.exceptions import APIError


@pytest.fixture
def client():
    """Create a test client."""
    return EvalClient(base_url="http://test.example.com")


def test_client_initialization():
    """Test client initialization."""
    client = EvalClient(base_url="http://test.example.com")
    assert client.base_url == "http://test.example.com"
    assert client.api_url == "http://test.example.com/api/v1"


@patch('requests.Session.request')
def test_health_check(mock_request, client):
    """Test health check endpoint."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}
    mock_request.return_value = mock_response
    
    result = client.health_check()
    
    assert result == {"status": "healthy"}
    mock_request.assert_called_once()
