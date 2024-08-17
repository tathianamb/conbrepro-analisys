import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning


logging.basicConfig(
    filename='STATISTICALANALYSIS/LOG-STATISTICALANALYSIS.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


def log_warning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{message} (Category: {category.__name__}, File: {filename}, Line: {lineno})')

warnings.showwarning = log_warning

# Filtrar warnings específicos
warnings.filterwarnings('always', category=InterpolationWarning)

def load_data(file_path):
    # Carregar dados do arquivo CSV
    logging.debug(f'Carregando dados do arquivo: {file_path}')
    data = pd.read_csv(file_path, parse_dates=[0], index_col=0, dayfirst=True)
    data.index = pd.to_datetime(data.index)
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


def seasonal_decompose_img(data, save_path):
    result = seasonal_decompose(data, model='additive')
    plt.figure(figsize=(12, 6))
    plt.title('Seasonal Decomposition')
    result.plot()
    plt.savefig(save_path, format='png')
    plt.close()


def save_acf_img(data, save_path):
    # Plotar ACF
    logging.debug(f'Plotando ACF e salvando a imagem em: {save_path}')
    plt.figure(figsize=(12, 6))
    plot_acf(data, lags=40)  # Ajuste o número de lags conforme necessário
    plt.title('ACF')
    plt.savefig(save_path, format='png')
    plt.close()

def save_pacf_img(data, save_path):
    # Plotar PACF
    logging.debug(f'Plotando PACF e salvando a imagem em: {save_path}')
    plt.figure(figsize=(12, 6))
    plot_pacf(data, lags=40)  # Ajuste o número de lags conforme necessário
    plt.title('PACF')
    plt.savefig(save_path, format='png')
    plt.close()


def create_save_images(data, file, save_folder_path):

    base_name = os.path.splitext(file)[0]
    plot_time_series(data[:150], base_name, os.path.join(save_folder_path, f"{base_name}_time_series.png"))
    seasonal_decompose_img(data[:150], os.path.join(save_folder_path, f"{base_name}_seasonal_decompose.png"))
    save_acf_img(data, os.path.join(save_folder_path, f"{base_name}_acf.png"))
    save_pacf_img(data, os.path.join(save_folder_path, f"{base_name}_pacf.png"))


def adf_test(data):
    logging.info('Teste Dickey-Fuller')
    adf_result = adfuller(data)
    logging.info(f'Statistic: {adf_result[0]}')
    logging.info(f'p-value: {adf_result[1]}')
    logging.info(f'Critical value: {adf_result[4]}')

    critical_values = adf_result[4]
    statistic = adf_result[0]
    for key, value in critical_values.items():
        if statistic < value:
            logging.info(
                f'ADF Statistic ({statistic}) é menor que o valor crítico de {key} ({value}) - Rejeita a hipótese nula de raiz unitária - é estacionária.')
        else:
            logging.info(
                f'ADF Statistic ({statistic}) é maior que o valor crítico de {key} ({value}) - Não rejeita a hipótese nula de raiz unitária - pode não ser estacionária.')


def kpss_test(data):
    logging.info('Teste KPSS')
    kpss_result = kpss(data, regression='c')
    logging.info(f'Statistic: {kpss_result[0]}')
    logging.info(f'p-value: {kpss_result[1]}')
    logging.info(f'Critical value: {kpss_result[3]}')

    critical_values = kpss_result[3]
    statistic = kpss_result[0]
    for key, value in critical_values.items():
        if statistic > value:
            logging.info(
                f'KPSS Statistic ({statistic}) é maior que o valor crítico de {key} ({value}) - Rejeita a hipótese nula de estacionaridade - pode não ser estacionária.')
        else:
            logging.info(
                f'KPSS Statistic ({statistic}) é menor que o valor crítico de {key} ({value}) - Não rejeita a hipótese nula de estacionaridade - é estacionária.')


def test_stationarity(data):
    logging.debug('Realizando o testes de estacionariedade')
    adf_test(data)
    kpss_test(data)


def log_statistics(data):
    logging.info(f'Quantidade de Observações: {len(data)}')
    logging.info(f'Data de Início: {data.index.min()}')
    logging.info(f'Data de Fim: {data.index.max()}')

    summary = data.describe()
    indentation = '                          '
    indented_summary = '\n'.join(f'{indentation}{line}' for line in str(summary).split('\n'))
    logging.info(f'Descrição estatística: \n{indented_summary}')

    test_stationarity(data)


load_folder_path = 'DATASET2023-INPUT'
files = [
    "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv",
    "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv"
]

save_folder_path = 'STATISTICALANALYSIS'

for file in files:
    file_path = os.path.join(load_folder_path, file)
    logging.info(f'Arquivo: {file}')
    data = load_data(file_path)

    create_save_images(data, file, save_folder_path)

    log_statistics(data)
