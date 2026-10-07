import pytest
from unittest import mock
from unittest.mock import MagicMock
import requests
from app.ut_practice import ThirdPartyServiceError, WeatherAlertService

# =====================================================================
# BLOCK 6: Network Mocking Practice (Third-Party HTTP Requests)
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Use 'unittest.mock.patch' to isolate the networking layer.
# - Real HTTP calls must be completely blocked during unit test execution.
# - Use 'MagicMock' to simulate HTTP 200, 404, or 500 status codes and raw JSON responses.
# - Use 'side_effect=requests.RequestException' to ensure proper handling of network timeout spikes.
# =====================================================================

class TestWeatherAlertService():

    @mock.patch.object(requests, 'get') # it is required to patch an object here to avoid incorrect target
    def test_status_code_200_with_red_alert_level(self, mock_get):
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "status": "success", "alert_level": "RED"}
        mock_get.return_value = response_mock
        assert service.should_pause_operations() is True

    @mock.patch.object(requests, 'get')
    def test_status_code_200_with_orange_alert_level(self, mock_get):
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "status": "success", "alert_level": "ORANGE"}
        mock_get.return_value = response_mock
        assert service.should_pause_operations() is True

    @mock.patch.object(requests, 'get')
    def test_status_code_200_with_wrong_alert_level(self, mock_get):
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "status": "success", "alert_level": "YELLOW"}
        mock_get.return_value = response_mock
        assert service.should_pause_operations() is False

    @mock.patch.object(requests, 'get')
    def test_weather_alert_service_with_empty_json(self, mock_get): # if response.json is empty
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {}
        mock_get.return_value = response_mock
        assert service.should_pause_operations() is False

    @mock.patch.object(requests, 'get')
    def test_upper_job_for_weather_alert_service(self, mock_get): # verifying .upper function
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "status": "success", "alert_level": "orange"}
        mock_get.return_value = response_mock
        assert service.should_pause_operations() is True


    @mock.patch.object(requests, "get", side_effect=ThirdPartyServiceError)  # raising ThirdPartyServiceError
    def test_raising_third_party_error_msg(self, mock_get):
        service = WeatherAlertService()
        with pytest.raises(ThirdPartyServiceError):
            service.should_pause_operations()

    @mock.patch.object(requests, "get")  # verifying error message for codes != 200
    def test_status_code_400(self, mock_get):
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 404
        mock_get.return_value = response_mock
        with pytest.raises(ThirdPartyServiceError) as e:
            service.should_pause_operations()
        assert "External service returned status code 404" in str(e.value)

    @mock.patch.object(requests, "get", side_effect=requests.exceptions.Timeout)  # verifying Timeout message
    def test_verifying_timeout_message_for_lost_connection(self, mock_get):
        service = WeatherAlertService()
        response_mock = MagicMock()
        response_mock.status_code = 500
        mock_get.return_value = response_mock
        with pytest.raises(ThirdPartyServiceError) as e:
            service.should_pause_operations()
        assert "Failed to connect to weather provider: " in str(e.value)
