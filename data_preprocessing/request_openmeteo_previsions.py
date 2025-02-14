import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta


def main():
    today = datetime.utcnow()
    start_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=6)).strftime('%Y-%m-%d')

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.0188874,
        "longitude": 18.1410136,
        "minutely_15": ["temperature_2m", "precipitation", "is_day", "shortwave_radiation", "global_tilted_irradiance"],
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "GMT",
        "start_date": start_date,
        "end_date": end_date,
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
    minutely_15_dataframe = minutely_15_dataframe[minutely_15_dataframe["is_day"] != 0.0]
    minutely_15_dataframe.to_csv('/app/datasets/minutely_dataframe_prevision.csv', index=False)

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.to_csv('/app/datasets/hourly_dataframe_prevision.csv', index=False)

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min

    daily_dataframe = pd.DataFrame(data=daily_data)
    daily_dataframe.to_csv('/app/datasets/daily_dataframe_prevision.csv', index=False)


if __name__ == "__main__":
    main()
