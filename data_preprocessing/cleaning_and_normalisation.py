import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


# 1 METODO PER SALVARE I DATI SU MONGODB
def save_data(df, output_path='/app/datasets_svr/'):
    df.to_csv(f'{output_path}training_dataset.csv', index=False)
    return "train, dev and test sets created"


# 2 METODO PER SELEZIONARE LE FEATURES NUMERICHE
def numerical_features(df):
    return df.select_dtypes(include=[np.number])


# 3 METODO PER LA SOSTITUZIONE DEI VALORI NULLI CON LA MEDIA DEI VALORI DELLA STESSA COLONNA
def null_values_substitution(df):
    # Creiamo una copia del DataFrame per evitare di modificare l'oggetto originale in modo indesiderato
    df_filled = df.copy()

    if 'state' in df_filled.columns:
        df_filled = df_filled.dropna(subset=['state'])

    for column in df_filled.select_dtypes(include='number').columns:
        mean_value = df_filled[column].mean()
        df_filled[column] = df_filled[column].fillna(mean_value)

    return df_filled


# 4 METODO PER L'IDENTIFICAZIONE E RIMOZIONE DI TUPLE CHE CONTENGONO OUTLIER
def remove_outliers(df):
    df_no_outliers = df.copy()

    for column in df_no_outliers.select_dtypes(include='number').columns:
        Q1 = df_no_outliers[column].quantile(0.25)
        Q3 = df_no_outliers[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        condition = (df_no_outliers[column] >= lower_bound) & (df_no_outliers[column] <= upper_bound)
        df_no_outliers = df_no_outliers[condition]

    df_no_outliers.reset_index(drop=True, inplace=True)

    return df_no_outliers


# 5 METODO CHE APPLICA LO SCALING ALLE FEATURES (SENZA LA COLONNA TARGET)
def scale_features_no_target(df, scaler_path='/app/data_preprocessing/feature_scaler.pkl'):
    df_scaled = df.copy()
    scaler = joblib.load(scaler_path)
    df_scaled[df_scaled.columns] = scaler.transform(df_scaled)

    return df_scaled


# 6 METODO CHE APPLICA LO SCALING ALLE FEATURES E ALLA COLONNA TARGET
def scale_features(df, target_column='state', method='minmax', scaler_path='/app/data_preprocessing/target_scaler.pkl'):
    df_scaled = df.copy()

    numeric_columns = df_scaled.select_dtypes(include='number').columns

    if method == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    elif method == 'zscore':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        raise ValueError("Metodo non valido. Usa 'minmax' o 'zscore'.")

    features = df_scaled.drop(columns=[target_column])
    target = df_scaled[[target_column]]
    df_scaled[features.columns] = feature_scaler.fit_transform(features)
    df_scaled[target_column] = target_scaler.fit_transform(target)

    # Salviamo il target scaler
    joblib.dump(target_scaler, scaler_path)
    joblib.dump(feature_scaler, '/app/data_preprocessing/feature_scaler.pkl')

    return df_scaled


# 7 METODO PER INVERTIRE LO SCALING
def inverse_scale_target(df, scaler_path='/app/data_preprocessing/target_scaler.pkl'):
    scaler = joblib.load(scaler_path)
    numeric_column = df.values.reshape(-1, 1)
    df_inverse_scaled = pd.DataFrame(scaler.inverse_transform(numeric_column), columns=[df.columns[0]])

    return df_inverse_scaled


# 8 METODO CHE CONSENTE DI CODIFICARE LE FEATURES CATEGORICHE COME VARIABILI DUMMY
def encode_categorical_features(df):
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['category']).columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=False)
    return df_encoded


# 9 METODO PER LA RIMOZIONE DELLE RIGHE DUPLICATE
def remove_duplicates(df):
    # Rimuoviamo le righe duplicate
    df_no_duplicates = df.drop_duplicates()

    return df_no_duplicates


# WRAPPERS:
def save_data_wrapper(df, output_path='/app/datasets_svr/'):
    message = save_data(df, output_path=output_path)
    return message


def numerical_features_wrapper(df, **kwargs):
    return numerical_features(df)


def null_values_substitution_wrapper(df, *args):
    return null_values_substitution(df)


def remove_outliers_wrapper(df, *args):
    return remove_outliers(df)


def scale_features_no_target_wrapper(df, scaler_path='/app/data_preprocessing/feature_scaler.pkl'):
    return scale_features_no_target(df, scaler_path=scaler_path)


def encode_categorical_features_wrapper(df, *args):
    return encode_categorical_features(df)


def remove_duplicates_wrapper(df, *args):
    return remove_duplicates(df)


def scale_features_wrapper(df, target_column='state', method='minmax',
                           scaler_path='/app/data_preprocessing/target_scaler.pkl'):
    return scale_features(df, target_column=target_column, method=method, scaler_path=scaler_path)


def main(operationType):
    preprocessing_pipeline_minutely = Pipeline([
        ('select_numerical', FunctionTransformer(numerical_features_wrapper, validate=False)),
        ('remove_duplicates', FunctionTransformer(remove_duplicates_wrapper, validate=False)),
        ('fill_nulls', FunctionTransformer(null_values_substitution_wrapper, validate=False)),
        ('remove_outliers', FunctionTransformer(remove_outliers_wrapper, validate=False)),
        ('encode_categorical', FunctionTransformer(encode_categorical_features_wrapper, validate=False)),
        ('scale_features', FunctionTransformer(scale_features_wrapper, validate=False, kw_args={'method': 'minmax'})),
        ('save_data',
         FunctionTransformer(save_data_wrapper, validate=False, kw_args={'output_path': '/app/datasets_svr/'}))
    ])

    preprocessing_pipeline_no_split = Pipeline([
        ('select_numerical', FunctionTransformer(numerical_features_wrapper, validate=False)),
        ('remove_duplicates', FunctionTransformer(remove_duplicates_wrapper, validate=False)),
        ('fill_nulls', FunctionTransformer(null_values_substitution_wrapper, validate=False)),
        ('encode_categorical', FunctionTransformer(encode_categorical_features_wrapper, validate=False)),
        ('scale_features_no_target', FunctionTransformer(scale_features_no_target_wrapper, validate=False))
    ])

    preprocessing_pipeline_to_final_dataset = Pipeline([
        ('remove_duplicates', FunctionTransformer(remove_duplicates_wrapper, validate=False)),
        ('fill_nulls', FunctionTransformer(null_values_substitution_wrapper, validate=False)),
        ('encode_categorical', FunctionTransformer(encode_categorical_features_wrapper, validate=False))
    ])

    if operationType == 'train':
        df2 = pd.read_csv("/app/datasets/combined_df_minutely_dataset.csv")

        df2['state'] = df2['state'].astype(float)

        preprocessing_pipeline_minutely.fit_transform(df2)

    elif operationType == 'prevision':
        df5 = pd.read_csv("/app/datasets/minutely_dataframe_prevision.csv")

        df_preprocessed5 = preprocessing_pipeline_no_split.fit_transform(df5)
        df_preprocessed5.to_csv('/app/datasets_svr/minutely_dataset_to_prediction.csv', index=False)
        df_preprocessed8 = preprocessing_pipeline_to_final_dataset.fit_transform(df5)
        df_preprocessed8.to_csv('/app/datasets_svr/minutely_dataset_to_final_dataset.csv', index=False)
