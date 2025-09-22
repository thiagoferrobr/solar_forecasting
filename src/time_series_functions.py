import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
import csv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

# --- Nova função auxiliar para encontrar colunas ---
def find_col_by_substring(columns, substring):
    """ Encontra o nome completo de uma coluna em uma lista a partir de uma substring. """
    for col in columns:
        if substring in col:
            return col
    return None

def load_multivariate_data(path, hour_min_max):
    """
    Carrega e pré-processa dados multivariados de forma robusta,
    encontrando as colunas por palavras-chave.
    """
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df_total.columns = [str(col).strip() for col in df_total.columns]
    
    # --- LÓGICA DE BUSCA POR PALAVRAS-CHAVE ---
    target_col_name = find_col_by_substring(df_total.columns, 'RADIACAO GLOBAL')
    temp_col_name = find_col_by_substring(df_total.columns, 'TEMPERATURA DO AR')
    umid_col_name = find_col_by_substring(df_total.columns, 'UMIDADE RELATIVA DO AR')
    vento_col_name = find_col_by_substring(df_total.columns, 'VENTO, VELOCIDADE')
    
    # Valida se todas as colunas essenciais foram encontradas
    required_cols = {'Irradiância': target_col_name, 'Temperatura': temp_col_name, 
                     'Umidade': umid_col_name, 'Vento': vento_col_name}
    for key, val in required_cols.items():
        if val is None:
            raise KeyError(f"Não foi possível encontrar a coluna de '{key}' no arquivo: {path}")
    
    feature_cols = [temp_col_name, umid_col_name, vento_col_name]
    # ----------------------------------------------

    df_total[target_col_name] = pd.to_numeric(df_total[target_col_name], errors='coerce')
    df_total[target_col_name] = (df_total[target_col_name] * 1000) / 3600 # W/m²
    
    for col in feature_cols:
        df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

    all_cols = [target_col_name] + feature_cols
    df_total[all_cols] = df_total[all_cols].ffill().bfill()

    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except (ValueError, TypeError):
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    
    df_total.rename(columns={target_col_name: 'actual'}, inplace=True)
    final_cols_renamed = ['actual'] + feature_cols
    df_total = df_total.set_index('Data')[final_cols_renamed]
        
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
