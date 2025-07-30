import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
from collections import deque
import matplotlib.pyplot as plt

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def create_windowing(df, lag_size):
    final_df = None
    for i in range(lag_size + 1):
        serie = df.shift(i)
        serie.columns = ['actual'] if i == 0 else [f'lag{i}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title, prevs_df=None):
    # Lógica de avaliação simplificada para o exemplo. A completa está no seu repo.
    test_metrics = {'RMSE': root_mean_square_error(y_true[-test_size:], y_pred[-test_size:])}
    geral_dict = {'test_metrics': test_metrics, 'params': model_params}
    if return_type == 3:
        save_result(geral_dict, title)
    return geral_dict.get('test_metrics', {})

def save_result(dict_result, title):
    title = f"{title}-{uuid.uuid4()}.pkl"
    with open(title, 'wb') as handle: pkl.dump(dict_result, handle)
    return title
    
def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = ts.shape[0] - test_size - val_size
    y_train = ts['actual'].iloc[:train_size]
    x_train = ts.drop(columns=['actual']).iloc[:train_size]
    return model.fit(x_train.values, y_train.values)

def predict_sklearn_model(ts, model):
    x = ts.drop(columns=['actual'])
    return model.predict(x.values)
    
def utc_hour_to_int(x): return int(str(x).split(' ')[0])

def load_data_solar_hours(path, min_max, use_log, save_cv):
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df_total.columns = df_total.columns.str.strip()
    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    
    if coluna_radiacao not in df_total.columns:
         raise KeyError(f"A coluna '{coluna_radiacao}' não foi encontrada. Colunas: {df_total.columns.tolist()}")

    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao], errors='coerce').fillna(0)
    df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(min_max[0]), utc_hour_to_int(min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total.loc[cond, ['Data', coluna_radiacao]]
    df_total = df_total.rename(columns={coluna_radiacao: 'actual'}).set_index('Data')
    
    if save_cv: df_total.to_csv(path.replace('.csv', '_solar.csv',-1))
    return df_total
