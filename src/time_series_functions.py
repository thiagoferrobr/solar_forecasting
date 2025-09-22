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
#               FUNÇÃO 1: PARA ANÁLISE UNIVARIADA (NOME ORIGINAL MANTIDO)
# ==============================================================================
def load_data_solar_hours(path, hour_min_max, use_log, save_cv):
    """ Carrega e limpa os dados especificamente para a análise univariada. """
    try:
        df = pd.read_csv(path, sep=';', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
        
    df.columns = [str(col).strip() for col in df.columns]
    
    coluna_radiacao_keyword = 'RADIACAO GLOBAL'
    coluna_radiacao = next((col for col in df.columns if coluna_radiacao_keyword in col), None)
    if not coluna_radiacao: raise KeyError(f"Coluna contendo '{coluna_radiacao_keyword}' não foi encontrada.")

    if df[coluna_radiacao].dtype == 'object':
        df[coluna_radiacao] = df[coluna_radiacao].str.replace(',', '.', regex=False)
    df[coluna_radiacao] = pd.to_numeric(df[coluna_radiacao], errors='coerce').fillna(0)

    df.rename(columns={coluna_radiacao: 'actual'}, inplace=True)
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
#               FUNÇÃO 2: PARA ANÁLISE MULTIVARIADA
# ==============================================================================
def find_col_by_substring(columns, substring):
    for col in columns:
        if substring in col: return col
    return None

def load_and_validate_data(path, hour_min_max):
    """ Carrega, limpa, valida features e retorna o dataframe e a lista de features válidas. """
    try:
        df = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df.columns = [str(col).strip() for col in df.columns]
    
    COLUMN_MAP = {'RADIACAO GLOBAL': 'actual', 'TEMPERATURA DO AR': 'temperatura',
                  'UMIDADE RELATIVA DO AR': 'umidade', 'VENTO, VELOCIDADE': 'vento_velocidade'}
    rename_dict = {find_col_by_substring(df.columns, k): v for k, v in COLUMN_MAP.items() if find_col_by_substring(df.columns, k)}
    df = df.rename(columns=rename_dict)
    
    all_possible_features = ['temperatura', 'umidade', 'vento_velocidade']
    expected_cols = ['actual'] + all_possible_features
    
    for col in expected_cols:
        if col not in df.columns: df[col] = np.nan

    for col in expected_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    valid_features = []
    for col in all_possible_features:
        if col in df.columns and df[col].notna().sum() / len(df) > 0.9 and df[col].std() > 0.1:
            valid_features.append(col)
    
    print(f"Features válidas encontradas neste arquivo: {valid_features if valid_features else 'Nenhuma'}")
    
    cols_to_use = ['actual'] + valid_features
    for col in cols_to_use:
        if df[col].isnull().any():
            if df[col].isnull().all(): df[col] = 0
            else: df[col] = df[col].ffill().bfill()
            
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

# ==============================================================================
#               FUNÇÕES AUXILIARES RESTANTES
# ==============================================================================
def create_multivariate_dataset(data, look_back=12):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title):
    y_pred_aligned = y_pred[~np.isnan(y_pred)]
    y_true_aligned = y_true[-len(y_pred_aligned):]
    test_metrics = {'RMSE': root_mean_square_error(y_true_aligned, y_pred_aligned)}
    geral_dict = {'test_metrics': test_metrics, 'params': model_params}
    if return_type == result_options.save_result: save_result(geral_dict, title)
    return geral_dict.get('test_metrics', {})

def save_result(dict_result, title):
    title = f"{title}-{uuid.uuid4()}.pkl"
    with open(title, 'wb') as handle: pkl.dump(dict_result, handle)
    return title

def open_saved_result(file_name):
    with open(file_name, 'rb') as handle: return pkl.load(handle)

def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = ts.shape[0] - test_size - val_size
    y_train = ts['actual'].iloc[:train_size]
    x_train = ts.drop(columns=['actual']).iloc[:train_size]
    return model.fit(x_train.values, y_train.values)

def predict_sklearn_model(ts, model):
    x = ts.drop(columns=['actual'])
    return model.predict(x.values)

print("Arquivo 'src/time_series_functions.py' sobrescrito com a versão final e completa.")