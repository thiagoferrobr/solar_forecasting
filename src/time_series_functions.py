import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
from collections import deque
import matplotlib.pyplot as plt

# Classe de opções de resultado
class result_options:
    test_result = 0
    val_result = 1
    train_result = 2
    save_result = 3

# Função para criar janelas de tempo
def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

# Funções de métricas de erro
def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)
    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):
    return mean_square_error(y_true, y_pred)**0.5

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    posi_with_zeros = np.where(y_true == 0)[0]
    y_true = [n for k, n in enumerate(y_true) if k not in posi_with_zeros]
    y_pred = [n for k, n in enumerate(y_pred) if k not in posi_with_zeros]
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return 0.0
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(np.abs(y_true - y_pred))

# Função para gerar o dicionário de métricas
def gerenerate_metric_results(y_true, y_pred):
    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE': root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            }

# Função principal de avaliação
def make_metrics_avaliation(y_true, y_pred, test_size,
                            val_size, return_type, model_params,
                            title, prevs_df=None):
    data_size = len(y_true)
    train_size = data_size - (val_size + test_size)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true_test = y_true[(data_size - test_size):data_size]
    y_pred_test = y_pred[(data_size - test_size):data_size]
    val_result = None
    if val_size > 0:
        y_true_val = y_true[(train_size):(data_size - test_size)]
        y_pred_val = y_pred[(train_size):(data_size - test_size)]
        val_result = gerenerate_metric_results(y_true_val, y_pred_val)
    y_true_train = y_true[:train_size]
    y_pred_train = y_pred[:train_size]
    geral_dict = {
        'test_metrics': gerenerate_metric_results(y_true_test, y_pred_test),
        'val_metrics': val_result,
        'train_metrics': gerenerate_metric_results(y_true_train, y_pred_train),
        'real_values': y_true,
        'predicted_values': y_pred,
        'pool_prevs': prevs_df,
        'params': model_params
    }
    if return_type == 0:
        return geral_dict['test_metrics']
    elif return_type == 1:
        return geral_dict['val_metrics']
    elif return_type == 2:
        return geral_dict['train_metrics']
    elif return_type == 3:
        return save_result(geral_dict, title)

# Funções para salvar e abrir resultados
def save_result(dict_result, title):
    title = title + "-" + str(uuid.uuid4()) + ".pkl"
    with open(title, 'wb') as handle:
        pkl.dump(dict_result, handle)
    return title

def open_saved_result(file_name):
    with open(file_name, 'rb') as handle:
        b = pkl.load(handle)
    return b

# Funções de treino e previsão do Scikit-learn
def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = ts.shape[0] - test_size - val_size
    y_train = ts['actual'][0:train_size]
    x_train = ts.drop(columns=['actual'], axis=1)[0:train_size]
    return model.fit(x_train.values, y_train.values)

def predict_sklearn_model(ts, model):
    x = ts.drop(columns=['actual'], axis=1)
    return model.predict(x.values)

# Função auxiliar para hora
def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

# VERSÃO FINAL E CORRIGIDA DA FUNÇÃO DE LEITURA DE DADOS
def load_data_solar_hours(path, min_max, use_log, save_cv):
    # Esta versão usa on_bad_lines='skip' para ser robusta contra arquivos mal formatados.
    df_total = pd.read_csv(
        path,
        sep=';',
        decimal=',',
        encoding='latin-1',
        skiprows=8,
        engine='python',
        on_bad_lines='skip'
    )
    
    df_total.columns = df_total.columns.str.strip()
    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    
    if coluna_radiacao not in df_total.columns:
         raise KeyError(f"A coluna '{coluna_radiacao}' não foi encontrada. Colunas disponíveis: {df_total.columns.tolist()}")

    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao], errors='coerce').fillna(0)
    df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    df_total.dropna(subset=['Data'], inplace=True)

    min_hour = utc_hour_to_int(min_max[0])
    max_hour = utc_hour_to_int(min_max[1])
    
    cond = df_total['Hora UTC'].apply(lambda x: utc_hour_to_int(x) >= min_hour and utc_hour_to_int(x) <= max_hour)
    
    df_total = df_total[cond][['Data', coluna_radiacao]]
    df_total.rename(columns={coluna_radiacao: 'actual'}, inplace=True)
    df_total = df_total.drop(columns=['Hora UTC'], errors='ignore')
    df_total.set_index('Data', inplace=True)
    
    if use_log:
        df_total['actual'] = np.log(df_total['actual'] + 1)
        
    if save_cv:
        df_total.to_csv(path.replace('.csv', '_solar.csv'))
        
    return df_total