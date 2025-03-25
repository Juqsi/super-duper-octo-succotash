import pytest
from unittest.mock import patch
from plant_api import PlantGetter, _PlantApi


def test_get_plant_data_success():
    """
    Tests _PlantAPI.get_plant_data with a valid plant name and a mocked API response.
    """
    mock_response = {"scientific_name": "Rosa indica", "family": "Rosaceae"}
    with patch.object(_PlantApi, 'get_plant_data', return_value=mock_response):
        with _PlantApi() as plant_api:
            response = plant_api.get_plant_data("Rosa indica")

    assert response == mock_response


def test_get_plant_data_failure():
    """
    Tests _PlantAPI.get_plant_data with a mocked API failure.
    """
    with patch.object(_PlantApi, 'get_plant_data', side_effect=ValueError("Invalid input")):
        with pytest.raises(ValueError, match="Invalid input"):
            with _PlantApi() as plant_api:
                plant_api.get_plant_data(123)  # Invalid input


def test_get_plant_list_data():
    """
    Tests PlantGetter.get_plant_list_data with multiple plant names and mocked API responses.
    """
    plant_names = ["Rosa indica", "Tulipa gesneriana"]
    mock_responses = [
        {"scientific_name": "Rosa indica", "family": "Rosaceae"},
        {"scientific_name": "Tulipa gesneriana", "family": "Liliaceae"}
    ]
    with patch.object(_PlantApi, 'get_plant_data', side_effect=mock_responses):
        results = PlantGetter.get_plant_list_data(plant_names)

    assert results == mock_responses