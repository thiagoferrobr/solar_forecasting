import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn import preprocessing
from tqdm import tqdm
import src.time_series_functions as tsf
from src.time_series_functions import root_mean_square_error

def get_windowing(ts_normalized, time_window, horizon, prefix=''):
    ts_windowed = tsf.create_windowing(df=ts_normalized, lag_size=(time_window + (horizon-1)))
    columns_lag = [f'lag_{l}{prefix}' for l in reversed(range(1, time_window + 1))]
    columns_horizon = [f'hor_{l}{prefix}' for l in range(1, horizon)] + ['actual']
    ts_windowed.columns = columns_lag + columns_horizon
    ts_windowed = ts_windowed[columns_lag + ['actual']]
    return ts_windowed

def single_model(title, type_data, time_window, time_series, model, test_size,
                 val_size, return_option, normalize, horizon=1, recursive=False, use_exo_future=True):
    train_size = len(time_series) - test_size - val_size
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
        min_max_scaler.fit(time_series['actual'].values[0:train_size].reshape(-1, 1))
        ts_normalized_values = min_max_scaler.transform(time_series['actual'].values.reshape(-1, 1))
        ts_normalized = pd.DataFrame({'actual': ts_normalized_values.flatten()})
    else:
        ts_normalized = time_series
    ts_windowed = get_windowing(ts_normalized, time_window, horizon)
    reg = tsf.fit_sklearn_model(ts_windowed, model, test_size, val_size)
    pred = tsf.predict_sklearn_model(ts_windowed, reg)
    pred_normalized = pred.copy()
    ts_atu_normalized = ts_normalized['actual'][-len(pred_normalized):].values
    ts_atu_unnormalized = time_series['actual'][-len(pred_normalized):].values
    pred_unnormalized = pred_normalized.copy()
    if normalize:
        pred_unnormalized = min_max_scaler.inverse_transform(pred_unnormalized.reshape(-1, 1)).flatten()
    # Descomente as linhas abaixo se quiser o output detalhado durante o GridSearch
    # rmse_unnormalized = root_mean_square_error(ts_atu_unnormalized, pred_unnormalized)
    # rmse_normalized = root_mean_square_error(ts_atu_normalized, pred_normalized)
    # print(f"\n  >> RMSE (Original / Não Normalizado): {rmse_unnormalized:.4f}")
    # print(f"  >> RMSE (Normalizado / Artigo):      {rmse_normalized:.4f}\n")
    results = tsf.make_metrics_avaliation(ts_atu_unnormalized, pred_unnormalized,
                                          test_size, val_size,
                                          return_option, model.get_params(deep=True),
                                          title + '(tw' + str(time_window) + ')')
    return results

def do_grid_search(type_data, real, test_size, val_size, parameters, model, horizon,
                   recurvise, use_exegen_future, model_execs):
    best_model, metric = None, 'RMSE'
    best_result = {'time_window': 0, metric: None}
    result_type = tsf.result_options.val_result
    list_params = list(ParameterGrid(parameters))
    for params in tqdm(list_params, desc='GridSearch'):
        params_actual = params.copy()
        if 'time_window' in params_actual: del params_actual['time_window']
        forecaster = clone(model).set_params(**params_actual)
        result_atual = [single_model('temp', type_data, params['time_window'], real, forecaster, test_size, val_size, result_type, True, horizon, recurvise, use_exegen_future)[metric] for _ in range(model_execs)]
        result = np.mean(np.array(result_atual))
        if best_result[metric] is None or best_result[metric] > result:
            best_model, best_result[metric], best_result['time_window'] = forecaster, result, params['time_window']
    return {'best_result': best_result, 'model': best_model}

# CÉLULA FINAL: ANÁLISE GRÁFICA E SUMÁRIO COMPLETO (ANO A ANO)

import glob
import pandas as pd
import os
from src import time_series_functions as tsf
import pickle

print("--- Gerando Análise Final e Sumário Comparativo Ano a Ano ---")

result_files = glob.glob('./solar_rad/**/*.pkl', recursive=True)
results_summary = []

for f_path in result_files:
    try:
        model_name_full = os.path.basename(os.path.dirname(f_path)).split('-', 1)[1]
        
        with open(f_path, 'rb') as f:
            result_data = pickle.load(f)

        # --- LÓGICA DE EXTRAÇÃO DO ANO CORRIGIDA ---
        # Lê o ano diretamente dos parâmetros salvos no arquivo .pkl
        params = result_data.get('params', {})
        year = params.get('year', 'Desconhecido') # Pega o ano ou 'Desconhecido' se não encontrar

        test_metrics = result_data.get('test_metrics', {})

        results_summary.append({
            'Ano': year,
            'Modelo': model_name_full.upper(),
            'RMSE': test_metrics.get('RMSE', float('nan')),
            'MAE': test_metrics.get('MAE', float('nan')),
            'R2': test_metrics.get('R2', float('nan'))
        })

    except Exception as e:
        print(f"Não foi possível processar o arquivo {f_path}: {e}")

# Gera o DataFrame consolidado com todos os resultados
if results_summary:
    df_summary = pd.DataFrame(results_summary)
    # Garante que a coluna 'Ano' seja numérica para ordenação
    df_summary['Ano'] = pd.to_numeric(df_summary['Ano'], errors='coerce')
    df_summary.dropna(subset=['Ano'], inplace=True)
    df_summary['Ano'] = df_summary['Ano'].astype(int)

    # --- TABELA 1: RESUMO ANO A ANO ---
    print("\n--- Tabela Comparativa Ano a Ano ---")
    df_yearly = df_summary.pivot_table(index='Modelo', columns='Ano', values='RMSE')
    print(df_yearly.to_string(formatters={col: '{:,.2f}'.format for col in df_yearly.columns}))

    # --- TABELA 2: RESUMO CONSOLIDADO POR MODELO ---
    print("\n\n--- Tabela Consolidada (Média Geral) ---")
    df_final_summary = df_summary.groupby('Modelo').mean().sort_values('RMSE')
    # Remove a coluna 'Ano' da média final
    df_final_summary = df_final_summary.drop(columns='Ano', errors='ignore')
    print(df_final_summary.to_string(formatters={
        'RMSE': '{:,.2f}'.format,
        'MAE': '{:,.2f}'.format,
        'R2': '{:,.3f}'.format
    }))

    # Salva os resultados para referência futura
    df_summary.to_csv('./analise_resultados_completa.csv', index=False)
else:
    print("Nenhum arquivo de resultado encontrado para gerar o sumário.")