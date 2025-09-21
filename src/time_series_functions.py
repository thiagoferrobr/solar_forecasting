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

def load_data_solar_hours(path, min_max, use_log, save_cv):
    # (Função univariada antiga, mantida para compatibilidade)
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    df_total.columns = [str(col).strip() for col in df_total.columns]
    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    if coluna_radiacao not in df_total.columns: raise KeyError(f"A coluna '{coluna_radiacao}' não foi encontrada.")
    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao], errors='coerce').fillna(0)
    df_total[coluna_radiacao] = (df_total[coluna_radiacao] * 1000) / 3600
    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except ValueError:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    df_total.dropna(subset=['Data'], inplace=True)
    min_hour, max_hour = utc_hour_to_int(min_max[0]), utc_hour_to_int(min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    df_total.rename(columns={coluna_radiacao: 'actual'}, inplace=True)
    df_total = df_total.drop(columns=['Hora UTC'], errors='ignore')
    df_total.set_index('Data', inplace=True)
    if save_cv: df_total.to_csv(path.replace('.csv', '_solar.csv'))
    return df_total

def load_multivariate_data(path, target_col, feature_cols, hour_min_max):
    """
    Carrega e pré-processa dados multivariados de um arquivo do INMET.
    """
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df_total.columns = [str(col).strip() for col in df_total.columns]
    
    if target_col not in df_total.columns:
         raise KeyError(f"A coluna alvo '{target_col}' não foi encontrada.")
    df_total[target_col] = pd.to_numeric(df_total[target_col], errors='coerce')
    df_total[target_col] = (df_total[target_col] * 1000) / 3600
    
    for col in feature_cols:
        if col in df_total.columns:
            df_total[col] = pd.to_numeric(df_total[col], errors='coerce')
        else:
            print(f"Aviso: A coluna de feature '{col}' não foi encontrada no arquivo.")

    all_cols = [target_col] + feature_cols
    df_total[all_cols] = df_total[all_cols].ffill().bfill()

    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='raise')
    except ValueError:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='coerce')
    
    df_total.dropna(subset=['Data'], inplace=True)

    # --- CORREÇÃO APLICADA AQUI ---
    # Usando o nome correto do parâmetro: hour_min_max
    min_hour, max_hour = utc_hour_to_int(hour_min_max[0]), utc_hour_to_int(hour_min_max[1])
    # -----------------------------

    cond = df_total['Hora UTC'].apply(lambda x: min_hour <= utc_hour_to_int(x) <= max_hour)
    df_total = df_total[cond]
    
    df_total.rename(columns={target_col: 'actual'}, inplace=True)
    final_cols = ['actual'] + feature_cols
    df_total = df_total.set_index('Data')[final_cols]
        
    return df_total

def create_multivariate_dataset(data, look_back=12):
    """
    Cria um dataset no formato de janela para previsão multivariada.
    """
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

# --- NOVA FUNÇÃO DE MÉTRICA ---
def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return r2_score(y_true, y_pred)

# --- FUNÇÃO DE MÉTRICAS ATUALIZADA ---
def gerenerate_metric_results(y_true, y_pred):
    # Garante que não haja NaNs que possam quebrar as métricas
    y_true_clean = np.nan_to_num(y_true)
    y_pred_clean = np.nan_to_num(y_pred)
    
    return {'RMSE': root_mean_square_error(y_true_clean, y_pred_clean),
            'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
            'R2': r_squared(y_true_clean, y_pred_clean)
           }

# --- NOVAS FUNÇÕES DE PLOTAGEM ---
def plot_predictions(y_true, y_pred, title='Comparativo Previsão vs. Real'):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(y_true, label='Valor Real', color='royalblue', linewidth=2)
    plt.plot(y_pred, label='Valor Previsto', color='darkorange', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Amostras de Teste', fontsize=12)
    plt.ylabel('Irradiância Solar (Kj/m²)', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def plot_feature_importance(model, features, title='Importância das Features'):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 8))
        plt.title(title)
        plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importância Relativa')
        plt.show()

def plot_daily_average(time_series, year):
    daily_avg = time_series['actual'].resample('D').mean()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 5))
    daily_avg.plot(label='Média Diária', color='teal')
    plt.title(f'Média Diária de Irradiância Solar - {year}', fontsize=16)
    plt.ylabel('Irradiância Solar Média (Kj/m²)')
    plt.xlabel('Data')
    plt.legend()
    plt.show()

def plot_monthly_variability(time_series, year):
    monthly_data = time_series.copy()
    monthly_data['Mês'] = monthly_data.index.month
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    monthly_data.boxplot(column='actual', by='Mês', grid=True)
    plt.title(f'Variabilidade Mensal da Irradiância - {year}', fontsize=16)
    plt.suptitle('') # Remove o título automático do pandas
    plt.xlabel('Mês')
    plt.ylabel('Irradiância Solar (Kj/m²)')
    plt.show()


