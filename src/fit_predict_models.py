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

def train_sklearn(model_execs, data_title, parameters, model):
    config_path, save_path = './', './solar_rad/'
    with open(f'{config_path}models_configuration_60_20_20.json') as f: data = json.load(f)
    recurvise, use_exegen_future, use_log = False, False, False
    for i in data:
        if i.get('activate', 0) == 1:
            print(i['name']); print(i['path_data'])
            test_size, val_size, type_data, horizon, min_max = i['test_size'], i['val_size'], i['type_data'], i['horzion'], i['hour_min_max']
            real = tsf.load_data_solar_hours(i['path_data'], min_max, use_log, False)
            gs_result = do_grid_search(type_data=type_data, real=real, test_size=test_size, val_size=val_size, parameters=parameters, model=model, horizon=horizon, recurvise=recurvise, use_exegen_future=use_exegen_future, model_execs=model_execs)
            print(gs_result)
            save_path_actual = f"{save_path}{type_data}-{data_title}/"
            os.makedirs(save_path_actual, exist_ok=True)
            title_temp = f"{type_data}-{data_title}"
            for _ in range(model_execs):
                single_model(f"{save_path_actual}{title_temp}", type_data, gs_result['best_result']['time_window'], real, gs_result['model'], test_size, val_size, tsf.result_options.save_result, True, horizon, recurvise, use_exegen_future)