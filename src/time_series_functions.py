import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

# --- FUNÇÃO DE MÉTRICAS ATUALIZADA ---
def gerenerate_metric_results(y_true, y_pred):
    y_true_clean = np.nan_to_num(y_true)
    y_pred_clean = np.nan_to_num(y_pred)
    return {
        'RMSE': root_mean_square_error(y_true_clean, y_pred_clean),
        'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
        'R2': r2_score(y_true_clean, y_pred_clean)
    }

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title):
    y_pred_aligned = y_pred[~np.isnan(y_pred)]
    y_true_aligned = y_true[-len(y_pred_aligned):]
    test_metrics = gerenerate_metric_results(y_true_aligned, y_pred_aligned)
    geral_dict = {
        'test_metrics': test_metrics, 
        'params': model_params,
        'real_values': y_true,
        'predicted_values': y_pred
    }
    if return_type == result_options.save_result:
        save_result(geral_dict, title)
    return geral_dict.get('test_metrics', {})

# ... (cole aqui o restante de TODAS as suas outras funções: load_and_validate_data, etc.) ...

print("Arquivo 'src/time_series_functions.py' atualizado com as novas métricas.")