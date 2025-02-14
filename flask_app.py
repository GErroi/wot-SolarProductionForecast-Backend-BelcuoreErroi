import json
import threading
import pandas as pd
from flask import Flask, jsonify
from pymongo import MongoClient
from data_preprocessing import request_openmeteo_previsions, cleaning_and_normalisation, request_openmeteo_train_dev_test
from machine_learning import supportVectorRegressor
from flask_cors import CORS
from upload_datasets import upload_data
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


@app.route('/uploadCsv', methods=['POST'])
def upload_csv():
    try:
        upload_data.forecasts_verification()
        request_openmeteo_previsions.main()
        cleaning_and_normalisation.main('prevision')
        supportVectorRegressor.main()

        return jsonify('upload complete'), 200
    except Exception as e:
        return jsonify(f'Errore: {str(e)}'), 500


@app.route('/downloadCsv', methods=['GET'])
def download_csv():
    try:
        client = MongoClient('mongodb://mongodb:27017/')
        db = client['SolarProductionForecast']
        collections = db.list_collection_names()
        data = {}

        for collection in collections:
            data[collection] = list(db[collection].find({}, {'_id': 0}))

        return jsonify(data)
    except Exception as e:
        return jsonify(f'Errore: {str(e)}'), 500


@app.route('/doTrain', methods=['POST'])
def train_models():
    try:
        thread = threading.Thread(target=background_train_models)
        thread.start()

        return jsonify('Training started'), 200
    except Exception as e:
        app.logger.error(f"Errore durante l'elaborazione: {e}")
        return jsonify(f'Errore: {str(e)}'), 500


@app.route('/evaluateResults', methods=['GET'])
def evaluate():
    try:
        data = '/app/datasets_svr/real_predict_evaluate.csv'
        data_df = pd.read_csv(data, header=None)
        result = [{"predicted": row[0], "real": row[1]} for row in data_df.iloc[:14, :2].values]

        return jsonify(result)

    except Exception as e:
        return jsonify(f'Errore: {str(e)}'), 500


@app.route('/trainingResults', methods=['GET'])
def traningResults():
    try:
        with open('/app/prediction/trainingResults.json') as f:
            training_results = json.load(f)

        return jsonify(training_results)
    except Exception as e:
        return jsonify(f'Errore: {str(e)}'), 500


def background_train_models():
    try:
        request_openmeteo_train_dev_test.main()
        cleaning_and_normalisation.main('train')
        supportVectorRegressor.train()
        app.logger.info("Training terminato.")
    except Exception as e:
        app.logger.error(f"Errore durante l'elaborazione: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
