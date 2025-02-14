import json
import pymongo
import csv
import os
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import logging
from multiprocessing import Lock

logging.basicConfig(level=logging.INFO)


def format_timestamp(timestamp):
    if isinstance(timestamp, pd.Timestamp):
        dt = timestamp.to_pydatetime()
    else:
        try:
            dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')
            except ValueError:
                print(f"Invalid timestamp format: {timestamp}")
                return timestamp

    dt += timedelta(hours=2)
    updated_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S%z')
    logging.info(f"timestamp originale: {timestamp}")
    logging.info(f"rimestap aggiornato: {updated_timestamp}")
    return updated_timestamp


def connect_to_mongodb(name_collection):
    client = pymongo.MongoClient("mongodb://mongodb:27017/")
    database_name = "SolarProductionForecast"
    db = client[database_name]
    collection = db[name_collection]
    return collection


def csv_to_json(csv_file):
    json_file = os.path.splitext(csv_file)[0] + '.json'
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    with open(json_file, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    print("Conversione completata. Il file JSON è stato creato con successo.")
    return json_file


def forecasts_verification():
    collection_name = 'forecasts_verification_daily'
    origin_collection = connect_to_mongodb('daily_prediction_final')
    forecasts_verification_collection = connect_to_mongodb(collection_name)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1) + timedelta(hours=23, minutes=59, seconds=59)
    seven_days_ago = today - timedelta(days=7)
    base_url = "https://gdfhome.duckdns.org/api/history/period/"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1MDdmNjliNjA0MmY0M2M1OTk0NjVmZDRiZmZlMWZjNyIsImlhdCI6MTcyNDc1MTMxMywiZXhwIjoyMDQwMTExMzEzfQ.S61REKbuTL1l4yP-iIQRDuZvyCmGWDJoL7-FQXAl7xg"
    }
    api_url = f"{base_url}{seven_days_ago.strftime('%Y-%m-%dT00:00:00Z')}?filter_entity_id=sensor.fimer_inverter_dc_power&end_time={yesterday.strftime('%Y-%m-%dT23:59:59Z')}"
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        history_data = response.json()
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return

    documents_to_copy = origin_collection.find()

    for doc in documents_to_copy:
        try:
            if isinstance(doc['date'], datetime):
                doc_date = doc['date']
            else:
                doc_date = datetime.strptime(doc['date'], '%Y-%m-%d %H:%M:%S%z')

            new_doc = {
                'date': doc_date,
                'last_updated': doc['last_updated'],
                'planned_state': doc['state'],
                'effective_state': take_effective_state(doc_date, history_data)
            }

            forecasts_verification_collection.update_one(
                {'date': doc_date},
                {'$set': new_doc},
                upsert=True
            )

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            continue

    forecasts_verification_collection.delete_many({
        'date': {'$lt': seven_days_ago}
    })

    documents = forecasts_verification_collection.find({
        'date': {'$lt': today},
        'effective_state': 'unavailable'
    })

    for doc in documents:
        try:
            if isinstance(doc['date'], datetime):
                doc_date = doc['date']
            else:
                doc_date = datetime.strptime(doc['date'], '%Y-%m-%d %H:%M:%S%z')
            new_effective_state = take_effective_state(doc_date, history_data)
            logging.info(f"New effective state: {new_effective_state}")

            forecasts_verification_collection.update_one(
                {'_id': doc['_id']},
                {'$set': {'effective_state': new_effective_state}}
            )

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            continue


def take_effective_state(date_ref, history_data):
    records = history_data[0] if history_data and len(history_data) > 0 else []
    if not records:
        return "unavailable"

    df = pd.DataFrame(records)
    processed_records = []
    for _, record in df.iterrows():
        if record['entity_id'] == 'sensor.fimer_inverter_dc_power':
            try:
                state_value = float(record['state']) if record['state'] != 'unavailable' else None
                processed_records.append({
                    'last_changed': pd.to_datetime(record['last_changed']),
                    'state': state_value
                })
            except (ValueError, TypeError):
                continue

    if not processed_records:
        return "unavailable"

    df_processed = pd.DataFrame(processed_records)
    df_processed['last_changed'] = df_processed['last_changed'].apply(format_timestamp)
    df_processed.set_index('last_changed', inplace=True)
    df_processed.index = pd.DatetimeIndex(df_processed.index)
    df_processed = df_processed.dropna()

    if df_processed.empty:
        return "unavailable"

    df_resampled = df_processed.resample('15min').first()
    date_ref = pd.to_datetime(date_ref)
    df_hourly = df_resampled['state'].multiply(0.25).resample('H').sum()
    df_daily = df_hourly.resample('D').sum()
    target_day = date_ref.floor('D')
    logging.info(f"Target day: {target_day}")
    logging.info(f"Index of df_daily: {df_daily.index}")

    if target_day.tzinfo is None:
        target_day = target_day.normalize().tz_localize('UTC')
    else:
        target_day = target_day.normalize().tz_convert('UTC')

    if df_daily.index.tzinfo is None:
        df_daily.index = df_daily.index.tz_localize('UTC')
    else:
        df_daily.index = df_daily.index.tz_convert('UTC')

    if target_day in df_daily.index:
        return df_daily[target_day]

    return "unavailable"


lock = Lock()


def main(csv_file):
    with lock:
        print("Processing started. It will take few minutes.")

        name_collection = os.path.splitext(os.path.basename(csv_file))[0]
        collection = connect_to_mongodb(name_collection)
        json_file = csv_to_json(csv_file)
        logging.info(json_file)

        with open(json_file, 'r') as file:
            data = json.load(file)

        for item in data:
            item['last_updated'] = datetime.now(timezone.utc)

        unique_values_in_csv = {item['date'] for item in data}

        collection.drop()
        for item in data:
            collection.insert_one(item)

        collection.delete_many({'date': {'$nin': list(unique_values_in_csv)}})
        print("Dati aggiornati con timestamp nel database MongoDB e dati obsoleti rimossi.")
