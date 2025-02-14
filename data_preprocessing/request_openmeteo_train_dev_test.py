from data_preprocessing import dataset_builder
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

def main():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    today = datetime.utcnow()
    end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.0188874,
        "longitude": 18.1410136,
        "start_date": "2023-09-22",
        "end_date": end_date,
        "minutely_15": ["temperature_2m", "precipitation", "is_day", "shortwave_radiation", "global_tilted_irradiance"],
        "timezone": "GMT",
        "tilt": 30,
        "azimuth": 45
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_precipitation = minutely_15.Variables(1).ValuesAsNumpy()
    minutely_15_is_day = minutely_15.Variables(2).ValuesAsNumpy()
    minutely_15_shortwave_radiation = minutely_15.Variables(3).ValuesAsNumpy()
    minutely_15_global_tilted_irradiance = minutely_15.Variables(4).ValuesAsNumpy()

    minutely_15_data = {"date": pd.date_range(
        start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=minutely_15.Interval()),
        inclusive="left"
    )}
    minutely_15_data["temperature_2m"] = minutely_15_temperature_2m
    minutely_15_data["precipitation"] = minutely_15_precipitation
    minutely_15_data["is_day"] = minutely_15_is_day
    minutely_15_data["shortwave_radiation"] = minutely_15_shortwave_radiation
    minutely_15_data["global_tilted_irradiance"] = minutely_15_global_tilted_irradiance


    minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
    minutely_15_dataframe.to_csv('/app/datasets/minutely_15_dataframe.csv', index=False)

    dataset_builder.main()


if __name__ == "__main__":
    main()
