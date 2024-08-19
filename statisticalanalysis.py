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


class StatisticalAnalysis:
    def __init__(self, load_folder_path, save_folder_path):
        self.load_folder_path = load_folder_path
        self.save_folder_path = save_folder_path
        self._ensure_save_folder_exists()

    def _ensure_save_folder_exists(self):
        if not os.path.exists(self.save_folder_path):
            os.makedirs(self.save_folder_path)
            logging.info(f'Criada a pasta para salvar arquivos: {self.save_folder_path}')

    def load_data(self, file_path):
        # Carregar dados do arquivo CSV
        logging.debug(f'Carregando dados do arquivo: {file_path}')
        data = pd.read_csv(file_path, parse_dates=[0], index_col=0, dayfirst=True)
        data.index = pd.to_datetime(data.index)
        return data

    def plot_time_series(self, data, title, save_path):
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

    def seasonal_decompose_img(self, data, save_path):
        result = seasonal_decompose(data, model='additive')
        plt.figure(figsize=(12, 6))
        plt.title('Seasonal Decomposition')
        result.plot()
        plt.savefig(save_path, format='png')
        plt.close()

    def save_acf_img(self, data, save_path):
        # Plotar ACF
        logging.debug(f'Plotando ACF e salvando a imagem em: {save_path}')
        plt.figure(figsize=(12, 6))
        plot_acf(data, lags=40)  # Ajuste o número de lags conforme necessário
        plt.title('ACF')
        plt.savefig(save_path, format='png')
        plt.close()

    def save_pacf_img(self, data, save_path):
        # Plotar PACF
        logging.debug(f'Plotando PACF e salvando a imagem em: {save_path}')
        plt.figure(figsize=(12, 6))
        plot_pacf(data, lags=40)  # Ajuste o número de lags conforme necessário
        plt.title('PACF')
        plt.savefig(save_path, format='png')
        plt.close()

    def create_save_images(self, data, file):
        base_name = os.path.splitext(file)[0]
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            logging.debug('Convertendo o índice para datetime.')
            data.index = data.index.to_timestamp()
        self.plot_time_series(data[:150], base_name, os.path.join(self.save_folder_path, f"{base_name}_time_series.png"))
        self.seasonal_decompose_img(data[:150], os.path.join(self.save_folder_path, f"{base_name}_seasonal_decompose.png"))
        self.save_acf_img(data, os.path.join(self.save_folder_path, f"{base_name}_acf.png"))
        self.save_pacf_img(data, os.path.join(self.save_folder_path, f"{base_name}_pacf.png"))

    def adf_test(self, data):
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

    def kpss_test(self, data):
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

    def test_stationarity(self, data):
        logging.debug('Realizando o testes de estacionariedade')
        self.adf_test(data)
        self.kpss_test(data)

    def log_statistics(self, data):
        logging.info(f'Quantidade de Observações: {len(data)}')
        logging.info(f'Data de Início: {data.index.min()}')
        logging.info(f'Data de Fim: {data.index.max()}')

        summary = data.describe()
        indentation = '                          '
        indented_summary = '\n'.join(f'{indentation}{line}' for line in str(summary).split('\n'))
        logging.info(f'Descrição estatística: \n{indented_summary}')

        self.test_stationarity(data)

    def process_files(self, files):
        if isinstance(files, pd.DataFrame):
            data = files
            file_name = os.path.splitext(data.name)[0]
            self.create_save_images(data, file_name)
            self.log_statistics(data)
        elif isinstance(files, list) and all(isinstance(file, str) for file in files):
            for file in files:
                file_path = os.path.join(self.load_folder_path, file)
                logging.info(f'Arquivo: {file}')
                data = self.load_data(file_path)
                self.create_save_images(data, file)
                self.log_statistics(data)
        else:
            raise ValueError("O argumento 'files' deve ser um DataFrame ou uma lista de caminhos de arquivos.")

# Exemplo de uso
if __name__ == "__main__":
    load_folder_path = 'DATASET2023-INPUT'
    save_folder_path = 'STATISTICALANALYSIS'
    files = [
        "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv",
        "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv",
        "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv",
        "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv"
    ]

    analysis = StatisticalAnalysis(load_folder_path, save_folder_path)
    analysis.process_files(files)
