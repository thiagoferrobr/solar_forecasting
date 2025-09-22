import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])
def load_multivariate_data(path, target_col, feature_cols, hour_min_max):
    """
    Carrega e pré-processa dados multivariados de um arquivo do INMET.
    """
    try:
        df_total = pd.read_csv(
            path, sep=';', decimal=',', encoding='latin-1', 
            skiprows=8, on_bad_lines='skip', engine='python'
        )
    except Exception:
        df_total = pd.read_csv(
            path, sep=',', encoding='latin-1', 
            skiprows=8, on_bad_lines='skip', engine='python'
        )
    
    df_total.columns = [str(col).strip() for col in df_total.columns]
    
    # Processa a coluna alvo (irradiância)
    if target_col not in df_total.columns:
         raise KeyError(f"A coluna alvo '{target_col}' não foi encontrada.")
    df_total[target_col] = pd.to_numeric(df_total[target_col], errors='coerce')
    df_total[target_col] = (df_total[target_col] * 1000) / 3600 # Converte para W/m²
    
    # Processa as colunas de features
    for col in feature_cols:
        if col in df_total.columns:
            df_total[col] = pd.to_numeric(df_total[col], errors='coerce')
        else:
            print(f"Aviso: A coluna de feature '{col}' não foi encontrada no arquivo.")

    # Trata dados faltantes nas features (preenchimento para frente e para trás)
    all_cols = [target_col] + feature_cols
    df_total[all_cols] = df_total[all_cols].ffill().bfill()

    # Processa data e hora
    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except ValueError:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(min_max[0]), utc_hour_to_int(min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    
    # Renomeia a coluna alvo e seleciona as colunas finais
    df_total.rename(columns={target_col: 'actual'}, inplace=True)
    final_cols = ['actual'] + feature_cols
    df_total = df_total.set_index('Data')[final_cols]
        
    return df_total


def create_multivariate_dataset(data, look_back=12):
    """
    Cria um dataset no formato de janela para previsão multivariada.
    
    Args:
        data (np.array): Array com os dados normalizados (coluna 0 deve ser o alvo).
        look_back (int): Janela de tempo.
        
    Returns:
        np.array, np.array: X (samples, timesteps, features), y (samples,)
    """
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        # Todas as colunas (features) da janela
        a = data[i:(i + look_back), :]
        dataX.append(a)
        # Apenas a coluna alvo (posição 0) do passo seguinte
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def load_data_solar_hours(path, hour_min_max, use_log, save_cv):
    # Função robusta para o caso univariado
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
    df['actual'] = (df['actual'] * 1000) / 3600 # W/m²
    try:
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except (ValueError, TypeError):
        df['Data'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    df.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    cond = df['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df = df[cond].set_index('Data')[['actual']]
    return df

def create_windowing(df, lag_size):
    final_df = None
    for i in range(lag_size + 1):
        serie = df.shift(i); serie.columns = ['actual'] if i == 0 else [f'lag{i}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title, prevs_df=None):
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
