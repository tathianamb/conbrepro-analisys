import os
import logging
import pandas as pd



class DataLoader:
    def __init__(self, file, folder_path):
        self.file_path = os.path.join(folder_path, file)

    def load_data(self, freq='h'):
        logging.info(f'Carregando dados do arquivo: {self.file_path}')
        try:
            data = pd.read_csv(self.file_path, parse_dates=[0], index_col=0, dayfirst=True)
            data.index = pd.DatetimeIndex(data.index).to_period(freq)
            logging.debug(f'Dados carregados com sucesso do arquivo: {self.file_path}')
        except Exception as e:
            logging.error(f'Erro ao carregar dados do arquivo: {self.file_path} - {e}')
            raise
        return data
