import joblib
from joblib import load
import numpy as np
import matplotlib as mpl
import pandas
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from data_preprocessing.cleaning_and_normalisation import inverse_scale_target, numerical_features
from upload_datasets import upload_data
import pandas as pd
import json

mpl.use('Agg')

DATASET_PATH = "/app/datasets_svr/"
filename = "/app/prediction/trainingResults.txt"


def predict_minutely():
    model = load("/app/models/minutely_model.sav")
    dataframe = pandas.read_csv('/app/datasets_svr/minutely_dataset_to_prediction.csv', sep=",", engine='python')
    future_dataframe = numerical_features(dataframe)
    output_file = '/app/prediction/minutely_prediction.csv'
    predict_and_save(model, future_dataframe, output_file)


def predict_hourly():
    dataframe = pandas.read_csv('/app/prediction/minutely_prediction_final.csv', sep=",", engine='python')
    dataframe['date'] = pandas.to_datetime(dataframe['date'])
    dataframe['date'] = dataframe['date'].dt.floor('H')
    produzione_oraria = dataframe.groupby('date')['state'].sum().reset_index()
    hourly_prevision = pandas.read_csv('/app/datasets/hourly_dataframe_prevision.csv')
    hourly_prevision['date'] = pandas.to_datetime(hourly_prevision['date'])
    merged_dataframe = pandas.merge(produzione_oraria, hourly_prevision, on='date', how='left')
    merged_dataframe.to_csv('/app/prediction/hourly_prediction_final.csv', index=False)
    upload_data.main('/app/prediction/hourly_prediction_final.csv')


def predict_daily():
    dataframe = pandas.read_csv('/app/prediction/hourly_prediction_final.csv', sep=",", engine='python')
    dataframe['date'] = pandas.to_datetime(dataframe['date'])
    dataframe['date'] = dataframe['date'].dt.floor('D')
    produzione_giornaliera = dataframe.groupby('date')['state'].sum().reset_index()
    daily_prevision = pandas.read_csv('/app/datasets/daily_dataframe_prevision.csv')
    daily_prevision['date'] = pandas.to_datetime(daily_prevision['date'])
    merged_dataframe = pandas.merge(produzione_giornaliera, daily_prevision, on='date', how='left')
    merged_dataframe.to_csv('/app/prediction/daily_prediction_final.csv', index=False)
    upload_data.main('/app/prediction/daily_prediction_final.csv')


def predict_and_save(model, future_dataframe, output_file):
    future_dataframe = generate_polynomial_features(future_dataframe, ['temperature_2m', 'shortwave_radiation', 'global_tilted_irradiance'], None)
    future_dataframe.dropna(inplace=True)
    future_data = future_dataframe.values
    future_data = future_data.reshape(future_data.shape[0], -1)
    pca = load("/app/models/pca_model.sav")
    future_data_pca = pca.transform(future_data)
    future_predictions = model.predict(future_data_pca)
    future_predictions_df = pd.DataFrame(future_predictions.reshape(-1, 1))
    future_predictions_df.columns = ["state"]
    future_predictions_df['state'] = np.abs(future_predictions_df['state'])
    future_predictions_orignal = inverse_scale_target(future_predictions_df,
                                                      scaler_path='/app/data_preprocessing/target_scaler.pkl')

    future_predictions_orignal.to_csv(output_file, index=False, header=True)
    print(f"Predictions of data production saved to {output_file}")

    minutely_dataset_to_prediction = pandas.read_csv('/app/datasets_svr/minutely_dataset_to_final_dataset.csv')
    minutely_dataset_to_prediction['state'] = future_predictions_orignal['state']
    combined_output_file = '/app/prediction/minutely_prediction_final.csv'
    minutely_dataset_to_prediction.to_csv(combined_output_file, index=False)

    upload_data.main('/app/prediction/minutely_prediction_final.csv')


def controlled_data_augmentation(dataframe):
    augmented_data = []
    for shift in range(1, 10):
        shifted = dataframe.shift(periods=shift).dropna()
        augmented_data.append(shifted)
    return pd.concat([dataframe] + augmented_data)


def add_gaussian_noise(dataframe, noise_factor=0.000000001):
    noisy_dataframe = dataframe.copy()
    noise = np.random.normal(0, noise_factor, dataframe.shape)
    noisy_dataframe.iloc[:, :] += noise
    return noisy_dataframe


def generate_polynomial_features(dataframe, columns=None, exclude_columns=None, degree=3):
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_columns:
        columns = [col for col in columns if col not in exclude_columns]

    for col in columns:
        for power in range(2, degree + 1):
            dataframe[f'{col}_poly_{power}'] = dataframe[col] ** power

    return dataframe


def train_dev_test_minutely():
    train_dataframe = pandas.read_csv(DATASET_PATH + 'training_dataset.csv', sep=",", engine='python')
    train_dataframe = controlled_data_augmentation(train_dataframe)
    train_dataframe = add_gaussian_noise(train_dataframe)
    train_dataframe = generate_polynomial_features(train_dataframe, ['temperature_2m', 'shortwave_radiation', 'global_tilted_irradiance'], "state")

    train_dataframe.dropna(inplace=True)

    train_data = train_dataframe.iloc[:, 1:].values
    train_labels = train_dataframe.iloc[:, 0].values

    X_train = train_data.reshape(train_data.shape[0], -1)
    Y_train = train_labels.ravel()

    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    joblib.dump(pca, '/app/models/pca_model.sav')

    # Definizione dei parametri da cercare per ogni stimatore
    # param_grid = {
    #     'rf__n_estimators': [50, 100, 200],
    #     'rf__max_depth': [None, 10, 20],
    #     'rf__min_samples_split': [2, 5],
    #
    #     'ridge__alpha': [0.1, 1.0, 10.0],
    #
    #     'kernel_ridge__alpha': [0.1, 1.0, 10.0],
    #     'kernel_ridge__kernel': ['rbf', 'polynomial'],
    #     'kernel_ridge__gamma': [0.1, 1.0, 'scale'],
    #
    #     'final_estimator__C': [1, 10, 100],
    #     'final_estimator__gamma': ['scale', 'auto', 0.1, 1.0],
    #     'final_estimator__epsilon': [0.01, 0.1],
    #     'final_estimator__degree': [2, 3]
    # }

    estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=100,
            random_state=28
        )),
        ('ridge', Ridge(
            alpha=1.0
        )),
        ('kernel_ridge', KernelRidge(
            kernel='rbf',
            alpha=1.0
        ))
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=SVR(
            kernel='rbf',
            gamma='scale',
            epsilon=0.01,
            degree=2,
            C=100
        ),
        cv=10
    )

    # Creazione del GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=stacking_regressor,
    #     param_grid=param_grid,
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=2,
    #     scoring='neg_mean_squared_error'
    # )

    kf = KFold(n_splits=10, shuffle=True, random_state=28)
    cv_rmse_scores = []
    all_predictions = []
    all_true_values = []

    for train_index, test_index in kf.split(X_train_pca):
        X_train_fold, X_test_fold = X_train_pca[train_index], X_train_pca[test_index]
        Y_train_fold, Y_test_fold = Y_train[train_index], Y_train[test_index]
        stacking_regressor.fit(X_train_fold, Y_train_fold)
        test_predictions = stacking_regressor.predict(X_test_fold)
        rmse = sqrt(mean_squared_error(
            inverse_scale_target(pd.DataFrame(Y_test_fold)).values.flatten(),
            inverse_scale_target(pd.DataFrame(test_predictions)).values.flatten()
        ))
        cv_rmse_scores.append(rmse)
        all_predictions.extend(test_predictions)
        all_true_values.extend(Y_test_fold)

    predictions_df = pd.DataFrame(all_predictions, columns=['predicted'])
    true_values_df = pd.DataFrame(all_true_values, columns=['true'])

    predictions_df_original = inverse_scale_target(predictions_df)
    true_values_df_original = inverse_scale_target(true_values_df)

    df_combined = pd.concat([predictions_df_original, true_values_df_original], axis=1)
    df_combined.columns = ['predicted', 'true']

    df_combined.to_csv('/app/datasets_svr/real_predict_evaluate.csv', index=False)

    cv_mean_rmse = np.mean(cv_rmse_scores)
    cv_std_rmse = np.std(cv_rmse_scores)

    stacking_regressor.fit(X_train_pca, Y_train)

    # Salvataggio risultati
    json_data = {
        "cross_validation_results": {
            "k_folds": 10,
            "mean_rmse": cv_mean_rmse,
            "std_rmse": cv_std_rmse,
            "individual_fold_rmse": cv_rmse_scores
        }
    }

    with open('/app/prediction/trainingResults.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    joblib.dump(stacking_regressor, '/app/models/minutely_model.sav')


def train():
    train_dev_test_minutely()


def main():
    predict_minutely()
    predict_hourly()
    predict_daily()
    return


if __name__ == '__main__':
    main()