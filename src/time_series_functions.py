import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
import csv

# ==============================================================================
#               VERSÃO COMPLETA E CORRIGIDA DE TODO O ARQUIVO
# ==============================================================================

# --- CORREÇÃO 1: Restaurada a classe 'result_options' completa ---
class result_options:
    test_result = 0
    val_result = 1
    train_result = 2
    save_result = 3

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

def load_data_solar_hours(path, min_max, use_log, save_cv):
    dados_extraidos = []
    colunas_finais = ['Data', 'Hora UTC', 'RADIACAO GLOBAL (Kj/m²)']

    with open(path, mode='r', encoding='latin-1') as f:
        for _ in range(8): next(f)
        next(f)
        for line in f:
            try:
                campos = line.strip().split(';')
                if len(campos) < 7: campos = line.strip().split(',')
                if len(campos) > 6:
                    dados_extraidos.append([campos[0], campos[1], campos[6]])
            except (IndexError, ValueError):
                continue

    if not dados_extraidos:
        raise ValueError(f"Nenhum dado válido pôde ser extraído do arquivo: {path}")

    df_total = pd.DataFrame(dados_extraidos, columns=colunas_finais)

    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao].str.replace(',', '.'), errors='coerce').fillna(0)
    
    # --- CORREÇÃO 2: Corrigido o formato da data para AAAA/MM/DD ---
    df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(min_max[0]), utc_hour_to_int(min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    df_total.rename(columns={coluna_radiacao: 'actual'}, inplace=True)
    df_total = df_total.drop(columns=['Hora UTC'], errors='ignore')
    df_total.set_index('Data', inplace=True)
    
    if save_cv:
        df_total.to_csv(path.replace('.csv', '_solar.csv'))
        
    return df_total

# Funções auxiliares (versões completas)
def create_windowing(df, lag_size):
    final_df = None
    for i in range(lag_size + 1):
        serie = df.shift(i)
        serie.columns = ['actual'] if i == 0 else [f'lag{i}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten())))
    
def gerenerate_metric_results(y_true, y_pred):
    return {'RMSE': root_mean_square_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred)}

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title, prevs_df=None):
    data_size = len(y_true)
    train_size = data_size - (val_size + test_size)
    y_true_test = y_true[-test_size:]
    y_pred_test = y_pred[-test_size:]
    
    test_metrics_results = gerenerate_metric_results(y_true_test, y_pred_test)
    
    geral_dict = {'test_metrics': test_metrics_results, 'params': model_params}
    
    if return_type == result_options.save_result:
        save_result(geral_dict, title)
    
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

print("Arquivo 'src/time_series_functions.py' sobrescrito com a versão final e 100% corrigida.")
