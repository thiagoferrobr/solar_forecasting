import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

# ==============================================================================
#               FUNÇÃO 1: APENAS PARA ANÁLISE UNIVARIADA
# ==============================================================================
def load_univariate_data(path, hour_min_max):
    """ Carrega e limpa os dados especificamente para a análise univariada. """
    df = pd.read_csv(
        path, sep=';', decimal=',', encoding='latin-1', 
        skiprows=8, on_bad_lines='skip', engine='python'
    )
    df.columns = [str(col).strip() for col in df.columns]
    
    coluna_radiacao = next((col for col in df.columns if 'RADIACAO GLOBAL' in col), None)
    if not coluna_radiacao: raise KeyError("Coluna 'RADIACAO GLOBAL' não encontrada.")

    df.rename(columns={coluna_radiacao: 'actual'}, inplace=True)
    df['actual'] = pd.to_numeric(df['actual'], errors='coerce').fillna(0)
    df['actual'] = (df['actual'] * 1000) / 3600 # Converte para W/m²

    try:
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except (ValueError, TypeError):
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
        
    df.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    
    df = df[cond].set_index('Data')[['actual']]
    return df

# ==============================================================================
#               FUNÇÃO 2: APENAS PARA ANÁLISE MULTIVARIADA
# ==============================================================================
def find_col_by_substring(columns, substring):
    for col in columns:
        if substring in col: return col
    return None

def load_multivariate_data(path, hour_min_max):
    """ Carrega, limpa e valida features para a análise multivariada. """
    df = pd.read_csv(
        path, sep=';', decimal=',', encoding='latin-1', 
        skiprows=8, on_bad_lines='skip', engine='python'
    )
    df.columns = [str(col).strip() for col in df.columns]
    
    COLUMN_MAP = {'RADIACAO GLOBAL': 'actual', 'TEMPERATURA DO AR': 'temperatura',
                  'UMIDADE RELATIVA DO AR': 'umidade', 'VENTO, VELOCIDADE': 'vento_velocidade'}
    rename_dict = {find_col_by_substring(df.columns, k): v for k, v in COLUMN_MAP.items() if find_col_by_substring(df.columns, k)}
    df = df.rename(columns=rename_dict)
    
    all_possible_features = ['temperatura', 'umidade', 'vento_velocidade']
    valid_features = []
    for col in all_possible_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            if df[col].notna().sum() / len(df) > 0.9 and df[col].std() > 0.1:
                valid_features.append(col)

    print(f"Features válidas encontradas: {valid_features if valid_features else 'Nenhuma'}")
    
    cols_to_use = ['actual'] + valid_features
    for col in cols_to_use:
        if df[col].isnull().any(): df[col] = df[col].ffill().bfill()
            
    df['actual'] = (df['actual'] * 1000) / 3600
    
    try:
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except (ValueError, TypeError):
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
        
    df.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    
    df = df[cond].set_index('Data')[cols_to_use]
    return df, valid_features
    
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
