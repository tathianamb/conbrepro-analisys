import os
import pandas as pd
import logging
from arma_model import ARMA_Model
#from lstm_model import LSTM_Model
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    filename='LOG-MODELLING.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        logging.debug(f'Carregando dados do arquivo: {self.file_path}')
        data = pd.read_csv(self.file_path, parse_dates=[0], index_col=0, dayfirst=True)
        data.index = pd.to_datetime(data.index)
        return data


if __name__ == "__main__":
    folder_path = 'DATASET2023-INPUT'
    files = [
        os.path.join(folder_path, "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv")
    ]

    df = DataLoader(files[0]).load_data()
    y = df['radiacao_global'].values

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

    train_size = int(len(y_scaled) * 0.6)
    val_size = int(len(y_scaled) * 0.2)

    y_train, y_temp = y_scaled[:train_size], y_scaled[train_size:]
    y_val, y_test = y_temp[:val_size], y_temp[val_size:]

    arma_model = ARMA_Model(y_train, y_val)
    arma_model, arma_order, arma_val_mse = arma_model.grid_search_arma()
    residuals = arma_model.resid
    print(residuals)