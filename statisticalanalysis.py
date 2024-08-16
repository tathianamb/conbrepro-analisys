import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def load_data(file_path):
    # Carregar dados do arquivo CSV
    data = pd.read_csv(file_path, parse_dates=[0], index_col=0, dayfirst=True)
    return data


def plot_time_series(data, title, save_path):
    # Plotar a série temporal
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
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')


def plot_acf(data, save_path_acf):
    # Plotar ACF
    plt.figure(figsize=(12, 6))
    plot_acf(data, ax=plt.gca())
    plt.title('ACF')
    plt.savefig(save_path_acf, format='png')
    plt.close()


def plot_pacf(data, save_path_pacf):
    # Plotar PACF
    plt.figure(figsize=(12, 6))
    plot_pacf(data, ax=plt.gca())
    plt.title('PACF')
    plt.savefig(save_path_pacf, format='png')
    plt.close()


# Caminho para os arquivos
load_folder_path = 'DATASET2023-INPUT'
files = [
    "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv",
    #"INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv",
    #"INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv"
]

save_folder_path = 'STATISTICALANALYSIS'
for file in files:
    file_path = os.path.join(load_folder_path, file)
    print(f'Analisando o arquivo: {file}')
    data = load_data(file_path)

    time_series_img = f"{file}_time_series.png"
    acf_img = f"{file}_acf.png"
    pacf_img = f"{file}_pacf.png"

    # Visualizar a série temporal e salvar a imagem
    plot_time_series(data, os.path.join(save_folder_path, file), time_series_img)

    # Testar a estacionariedade
    print("Teste de Estacionariedade:")
    test_stationarity(data)

    # Plotar ACF e salvar a imagem
    print("Plotando ACF:")
    plot_acf(data, acf_img)

    # Plotar PACF e salvar a imagem
    print("Plotando PACF:")
    plot_pacf(data, pacf_img)
