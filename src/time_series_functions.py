import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json

class result_options:
    test_result, val_result, train_result, save_result = 0, 1, 2, 3


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

def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

def load_data_solar_hours(path, min_max, use_log, save_cv):
    try:
        df_total = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    except Exception:
        df_total = pd.read_csv(path, sep=',', decimal=',', encoding='latin-1', skiprows=8, on_bad_lines='skip', engine='python')
    
    df_total.columns = [str(col).strip() for col in df_total.columns]
    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    
    if coluna_radiacao not in df_total.columns:
        raise KeyError(f"A coluna '{coluna_radiacao}' não foi encontrada. Colunas disponíveis: {df_total.columns.tolist()}")

    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao], errors='coerce').fillna(0)
    
    # Tenta múltiplos formatos de data para máxima compatibilidade
    try:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%Y/%m/%d %H%M UTC', errors='raise')
    except ValueError:
        df_total['Data'] = pd.to_datetime(df_total['Data'] + ' ' + df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC', errors='coerce')
    
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

def create_windowing(df, lag_size):
    final_df = None
    for i in range(lag_size + 1):
        serie = df.shift(i); serie.columns = ['actual'] if i == 0 else [f'lag{i}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title, prevs_df=None):
    test_metrics = {'RMSE': root_mean_square_error(y_true[-test_size:], y_pred[-test_size:])}
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

print("Arquivo 'src/time_series_functions.py' sobrescrito com a versão final.")
