import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import uuid
import json
import csv # Importa a biblioteca para leitura manual de CSV

# As outras funções do arquivo são mantidas
def utc_hour_to_int(x):
    return int(str(x).split(' ')[0])

# =================================================================================
#               VERSÃO FINAL COM LEITURA MANUAL E ROBUSTA
# =================================================================================
def load_data_solar_hours(path, min_max, use_log, save_cv):
    dados_extraidos = []
    colunas_finais = ['Data', 'Hora UTC', 'RADIACAO GLOBAL (Kj/m²)']

    with open(path, mode='r', encoding='latin-1') as f:
        # Pula as 8 linhas de metadados
        for _ in range(8):
            next(f)
        
        # Pula a linha de cabeçalho
        next(f)

        # Processa o restante do arquivo, linha por linha
        for line in f:
            try:
                # Tenta separar a linha por ponto e vírgula, que é o mais provável
                campos = line.strip().split(';')
                
                # Se a linha tiver colunas suficientes, extrai os dados pela posição
                if len(campos) > 6:
                    data = campos[0]
                    hora = campos[1]
                    radiacao = campos[6] # Coluna G (índice 6)
                    
                    dados_extraidos.append([data, hora, radiacao])
            except (IndexError, ValueError):
                # Se qualquer erro ocorrer ao processar a linha, ignora e continua
                continue

    if not dados_extraidos:
        raise ValueError("Nenhum dado válido pôde ser extraído do arquivo. Verifique o formato.")

    # Cria um DataFrame limpo a partir dos dados extraídos manualmente
    df_total = pd.DataFrame(dados_extraidos, columns=colunas_finais)

    # O resto da função continua a partir daqui, com um DataFrame garantido e limpo
    coluna_radiacao = 'RADIACAO GLOBAL (Kj/m²)'
    df_total[coluna_radiacao] = pd.to_numeric(df_total[coluna_radiacao].str.replace(',', '.'), errors='coerce').fillna(0)
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

# O resto das funções do arquivo (create_windowing, make_metrics_avaliation, etc.)
# devem ser mantidas aqui para que o código continue funcionando.
# Esta célula já contém uma versão simplificada delas.
def create_windowing(df, lag_size):
    final_df = None; serie = None
    for i in range(lag_size + 1):
        serie = df.shift(i)
        serie.columns = ['actual'] if i == 0 else [f'lag{i}']
        final_df = pd.concat([serie, final_df], axis=1)
    return final_df.dropna()

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.subtract(np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()))))

def make_metrics_avaliation(y_true, y_pred, test_size, val_size, return_type, model_params, title, prevs_df=None):
    test_metrics = {'RMSE': root_mean_square_error(y_true[-test_size:], y_pred[-test_size:])}
    geral_dict = {'test_metrics': test_metrics, 'params': model_params}
    class result_options: save_result = 3
    if return_type == result_options.save_result:
        save_result(geral_dict, title)
    return geral_dict.get('test_metrics', {})

def save_result(dict_result, title):
    title = f"{title}-{uuid.uuid4()}.pkl"
    with open(title, 'wb') as handle: pkl.dump(dict_result, handle)
    return title
    
def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = ts.shape[0] - test_size - val_size
    y_train = ts['actual'].iloc[:train_size]
    x_train = ts.drop(columns=['actual']).iloc[:train_size]
    return model.fit(x_train.values, y_train.values)

def predict_sklearn_model(ts, model):
    x = ts.drop(columns=['actual'])
    return model.predict(x.values)

print("Arquivo 'src/time_series_functions.py' sobrescrito com a versão final e robusta.")
