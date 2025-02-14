import pandas as pd


def main():
    df1 = pd.read_csv('/app/datasets/solar_production_full.csv')
    df3 = pd.read_csv('/app/datasets/minutely_15_dataframe.csv')

    df1['last_changed'] = pd.to_datetime(df1['last_changed']).dt.tz_localize(None)
    df3['date'] = pd.to_datetime(df3['date']).dt.tz_localize(None)

    df1_hourly = df1[(df1['last_changed'].dt.minute == 0) & (df1['last_changed'].dt.second == 0) & (
            df1['last_changed'].dt.microsecond == 0)].copy()

    df1_random = df1[~((df1['last_changed'].dt.minute == 0) & (df1['last_changed'].dt.second == 0) & (
            df1['last_changed'].dt.microsecond == 0))].copy()

    df1_random.to_csv('/app/datasets/solar_production_partial.csv', index=False)

    ######## CREA DATASET CON INFORMAZIONI OGNI 15 MINUTI##########
    df = pd.read_csv('/app/datasets/solar_production_partial.csv', header=None,
                     names=['entity_id', 'state', 'last_changed'])
    df['last_changed'] = pd.to_datetime(df['last_changed'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['last_changed'])
    df.set_index('last_changed', inplace=True)

    df_resampled = df.resample('15min').first().dropna().reset_index()

    df_resampled.to_csv('/app/datasets/resampled_data.csv', index=False, header=['last_changed', 'entity_id', 'state'])

    df5 = pd.read_csv('/app/datasets/resampled_data.csv')

    df5['last_changed'] = pd.to_datetime(df5['last_changed']).dt.tz_localize(None)
    df5 = df5.rename(columns={'last_changed': 'date'})

    combined_df_minutely = pd.merge(df5, df3, on='date', how='outer')

    cols = combined_df_minutely.columns.tolist()
    cols.insert(0, cols.pop(cols.index('date')))
    combined_df_minutely = combined_df_minutely[cols]

    combined_df_minutely = combined_df_minutely.fillna('null')
    combined_df_minutely['state'] = combined_df_minutely['state'].replace('unavailable', 'null')
    combined_df_minutely['state'] = pd.to_numeric(combined_df_minutely['state'], errors='coerce')

    combined_df_minutely['state'] = combined_df_minutely['state'] * 0.25
    combined_df_minutely['state'] = combined_df_minutely['state'].fillna('null')

    combined_df_minutely.to_csv('/app/datasets/combined_df_minutely_dataset.csv', index=False)
    ###############################################################

    print("Dati resampled salvati in 'combined_df_minutely_dataset.csv'")
