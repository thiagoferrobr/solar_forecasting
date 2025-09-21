import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
import csv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==============================================================================
#               VERSÃO FINAL COMPLETA DE TODAS AS FUNÇÕES
# ==============================================================================

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

def load_multivariate_data(path, target_col, feature_cols, hour_min_max):
    # (Função multivariada que já corrigimos)
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    df_total.columns = [str(col).strip() for col in df_total.columns]
    if target_col not in df_total.columns: raise KeyError(f"A coluna alvo '{target_col}' não foi encontrada.")
    df_total[target_col] = pd.to_numeric(df_total[target_col], errors='coerce')
    df_total[target_col] = (df_total[target_col] * 1000) / 3600
    for col in feature_cols:
        if col in df_total.columns: df_total[col] = pd.to_numeric(df_total[col], errors='coerce')
    all_cols = [target_col] + feature_cols
    df_total[all_cols] = df_total[all_cols].ffill().bfill()
    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except ValueError:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    df_total.rename(columns={target_col: 'actual'}, inplace=True)
    final_cols = ['actual'] + feature_cols
    df_total = df_total.set_index('Data')[final_cols]
    return df_total

def create_multivariate_dataset(data, look_back=12):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten())))

def r_squared(y_true, y_pred):
    return r2_score(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten())

def gerenerate_metric_results(y_true, y_pred):
    y_true_clean = np.nan_to_num(y_true)
    y_pred_clean = np.nan_to_num(y_pred)
    return {'RMSE': root_mean_square_error(y_true_clean, y_pred_clean),
            'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
            'R2': r_squared(y_true_clean, y_pred_clean)}

# --- FUNÇÃO make_metrics_avaliation INCLUÍDA AQUI ---
def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title):
    # Alinhamento dos vetores de previsão
    y_true_test = y_true[-len(y_pred):] # Garante que y_true tenha o mesmo tamanho de y_pred
    
    # Calcula as métricas apenas no conjunto de teste
    test_metrics = gerenerate_metric_results(y_true_test, y_pred)
    
    # Adiciona os vetores completos e os parâmetros para salvamento
    geral_dict = {
        'test_metrics': test_metrics, 
        'params': model_params,
        'real_values': y_true,
        'predicted_values': y_pred
    }
    
    if return_type == result_options.save_result:
        save_result(geral_dict, title)
        
    return geral_dict.get('test_metrics', {})
# ----------------------------------------------------

def save_result(dict_result, title):
    title = f"{title}-{uuid.uuid4()}.pkl"
    with open(title, 'wb') as handle: pkl.dump(dict_result, handle)
    return title

def open_saved_result(file_name):
    with open(file_name, 'rb') as handle: return pkl.load(handle)

print("Arquivo 'src/time_series_functions.py' sobrescrito com a versão final e completa.")
