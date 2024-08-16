import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import logging

# Configuração do logging
log_file_path = 'STATISTICALANALYSIS/LOG-STATISTICALANALYSIS.txt'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Cria o diretório se não existir

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('matplotlib').setLevel(logging.ERROR)

def load_data(file_path):
    # Carregar dados do arquivo CSV
    logging.debug(f'Carregando dados do arquivo: {file_path}')
    data = pd.read_csv(file_path, parse_dates=[0], index_col=0, dayfirst=True)
    return data

def plot_time_series(data, title, save_path):
    # Plotar a série temporal
    logging.debug(f'Plotando a série temporal e salvando a imagem em: {save_path}')
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Valor')
    plt.title(f'Série Temporal - {title}')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, format='png')
    plt.close()

def test_stationarity(data):
    # Testar a estacionariedade usando o teste de Dickey-Fuller
    logging.debug('Realizando o teste de estacionariedade (Dickey-Fuller)')
    result = adfuller(data)
    logging.info(f'ADF Statistic: {result[0]}')
    logging.info(f'p-value: {result[1]}')
    logging.info(f'Critical Values: {result[4]}')

def save_acf_img(data, save_path_acf):
    # Plotar ACF
    logging.debug(f'Plotando ACF e salvando a imagem em: {save_path_acf}')
    plt.figure(figsize=(12, 6))
    plot_acf(data, lags=40)  # Ajuste o número de lags conforme necessário
    plt.title('ACF')
    plt.savefig(save_path_acf, format='png')
    plt.close()

def save_pacf_img(data, save_path_pacf):
    # Plotar PACF
    logging.debug(f'Plotando PACF e salvando a imagem em: {save_path_pacf}')
    plt.figure(figsize=(12, 6))
    plot_pacf(data, lags=40)  # Ajuste o número de lags conforme necessário
    plt.title('PACF')
    plt.savefig(save_path_pacf, format='png')
    plt.close()

# Caminho para os arquivos
load_folder_path = 'DATASET2023-INPUT'
files = [
    "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv"
]

save_folder_path = 'STATISTICALANALYSIS'
os.makedirs(save_folder_path, exist_ok=True)  # Cria o diretório se não existir

for file in files:
    file_path = os.path.join(load_folder_path, file)
    logging.info(f'Analisando o arquivo: {file}')
    data = load_data(file_path)

    # Remove a extensão do arquivo CSV para criar nomes de imagens
    base_name = os.path.splitext(file)[0]

    time_series_img_path = os.path.join(save_folder_path, f"{base_name}_time_series.png")
    acf_img_path = os.path.join(save_folder_path, f"{base_name}_acf.png")
    pacf_img_path = os.path.join(save_folder_path, f"{base_name}_pacf.png")

    # Visualizar a série temporal e salvar a imagem
    plot_time_series(data[:150], base_name, time_series_img_path)

    # Testar a estacionariedade
    test_stationarity(data)

    # Plotar ACF e salvar a imagem
    save_acf_img(data, acf_img_path)

    # Plotar PACF e salvar a imagem
    save_pacf_img(data, pacf_img_path)
