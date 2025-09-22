import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
import csv
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
#               ARQUIVO FINAL COM FUNÇÃO DE LEITURA SUPER ROBUSTA
# ==============================================================================

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

def find_and_rename_columns(df):
    COLUMN_MAP = {
        'RADIACAO GLOBAL': 'actual', 'TEMPERATURA DO AR': 'temperatura',
        'UMIDADE RELATIVA DO AR': 'umidade', 'VENTO, VELOCIDADE': 'vento_velocidade'
    }
    rename_dict = {}
    for keyword, std_name in COLUMN_MAP.items():
        for col_name in df.columns:
            if keyword in col_name:
                rename_dict[col_name] = std_name; break
    return df.rename(columns=rename_dict)

def load_and_clean_data(path, hour_min_max):
    try:
        df = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df = pd.read_csv(path, sep=',', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df.columns = [str(col).strip() for col in df.columns]
    df = find_and_rename_columns(df)
    
    expected_cols = ['actual', 'temperatura', 'umidade', 'vento_velocidade']
    for col in expected_cols:
        if col not in df.columns: raise KeyError(f"Coluna padrão '{col}' não encontrada.")
    
    # --- CORREÇÃO DEFINITIVA NO TRATAMENTO NUMÉRICO ---
    for col in expected_cols:
        # Primeiro, garante que a coluna é do tipo string e substitui vírgula por ponto
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        # Depois, converte para numérico
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # --------------------------------------------------

    df[expected_cols] = df[expected_cols].ffill().bfill()
    df['actual'] = (df['actual'] * 1000) / 3600 # W/m²
    
    try:
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except (ValueError, TypeError):
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    
    df.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df = df[cond]
    
    df = df.set_index('Data')[expected_cols]
    return df
    
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
