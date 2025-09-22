import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn import preprocessing
from tqdm import tqdm
import src.time_series_functions as tsf
from src.time_series_functions import root_mean_square_error

# ==============================================================================
#               ARQUIVO COMPLETO E CORRIGIDO
# ==============================================================================

def get_windowing(ts_normalized, time_window, horizon, prefix=''):
    # ... (código da função inalterado) ...
    ts_windowed = tsf.create_windowing(df=ts_normalized, lag_size=(time_window + (horizon-1)))
    columns_lag = [f'lag_{l}{prefix}' for l in reversed(range(1, time_window + 1))]
    columns_horizon = [f'hor_{l}{prefix}' for l in range(1, horizon)] + ['actual']
    ts_windowed.columns = columns_lag + columns_horizon
    ts_windowed = ts_windowed[columns_lag + ['actual']]
    return ts_windowed

def single_model(title, type_data, time_window, time_series, model, test_size,
                 val_size, return_option, normalize, horizon=1, recursive=False, use_exo_future=True):
    # ... (código da função inalterado) ...
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
    ts_atu_unnormalized = time_series['actual'].values[-len(pred_normalized):]
    if normalize:
        pred_unnormalized = min_max_scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
    else:
        pred_unnormalized = pred_normalized.copy()
    params_to_save = model.get_params(deep=True)
    try:
        year = title.split('_')[-1].split('(')[0]
        params_to_save['year'] = year
    except: pass
    params_to_save['model_object'] = reg
    results = tsf.make_metrics_avaliation(
        ts_atu_unnormalized, pred_unnormalized,
        test_size, val_size,
        return_option, params_to_save, title
    )
    return results

def do_grid_search(type_data, real, test_size, val_size, parameters, model, horizon,
                   recurvise, use_exegen_future, model_execs):
    # ... (código da função inalterado) ...
    best_model, metric = None, 'RMSE'
    best_result = {'time_window': 0, metric: None}
    result_type = tsf.result_options.val_result
    list_params = list(ParameterGrid(parameters))
    for params in tqdm(list_params, desc='GridSearch'):
        params_actual = params.copy()
        time_window = params_actual.pop('time_window', 12)
        forecaster = clone(model).set_params(**params_actual)
        result_atual = [single_model('temp', type_data, time_window, real, forecaster, test_size, val_size, result_type, True, horizon, recurvise, use_exegen_future)[metric] for _ in range(model_execs)]
        result = np.mean(np.array(result_atual))
        if best_result[metric] is None or best_result[metric] > result:
            best_model, best_result[metric], best_result['time_window'] = forecaster, result, time_window
    return {'best_result': best_result, 'model': best_model}

def train_sklearn(model_execs, data_title, parameters, model):
    config_path, save_path = './', './solar_rad/'
    with open(f'{config_path}models_configuration_60_20_20.json') as f: data = json.load(f)
    for i in data:
        if i.get('activate', 0) == 1:
            print(f"\nProcessando: {i['name']}")

            # --- CORREÇÃO APLICADA AQUI ---
            # Voltamos a chamar a função correta: load_data_solar_hours
            real = tsf.load_data_solar_hours(i['path_data'], i['hour_min_max'], use_log=False, save_cv=False)
            # -----------------------------
            
            gs_result = do_grid_search(type_data=i['type_data'], real=real, test_size=i['test_size'],
                                       val_size=i['val_size'], parameters=parameters, model=model,
                                       horizon=i['horzion'], recurvise=False, use_exegen_future=False,
                                       model_execs=model_execs)
            
            print(gs_result)
            print(f"Melhores parâmetros encontrados para {data_title}: {gs_result['model'].get_params()}")
            
            save_path_actual = f"{save_path}{i['type_data']}-{data_title}/"
            os.makedirs(save_path_actual, exist_ok=True)
            title_temp = f"{i['type_data']}-{data_title}_{i['name'].split('_')[-1]}"
            
            single_model(f"{save_path_actual}{title_temp}", i['type_data'], gs_result['best_result']['time_window'],
                         real, gs_result['model'], i['test_size'], i['val_size'],
                         tsf.result_options.save_result, True, i['horzion'])

print("Arquivo 'src/fit_predict_models.py' restaurado e corrigido com sucesso.")