import os
import pandas as pd
import logging
from arma_model import ARMA_Model
#from lstm_model import LSTM_Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
        logging.info(f'Carregando dados do arquivo: {self.file_path}')
        try:
            data = pd.read_csv(self.file_path, parse_dates=[0], index_col=0, dayfirst=True)
            data.index = pd.to_datetime(data.index)
            logging.debug(f'Dados carregados com sucesso do arquivo: {self.file_path}')
        except Exception as e:
            logging.error(f'Erro ao carregar dados do arquivo: {self.file_path} - {e}')
            raise
        return data


if __name__ == "__main__":
    folder_path = 'DATASET2023-INPUT'
    files = [
        os.path.join(folder_path, "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_processed.csv"),
        os.path.join(folder_path, "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_processed.csv")
    ]

    try:
        df = DataLoader(files[0]).load_data()
        y = df['radiacao_global'].values
        logging.info(f'Dados de radiacao_global extraídos com sucesso.')

        y_train, y_temp = train_test_split(y, train_size=0.6, shuffle=False)
        y_val, y_test = train_test_split(y_temp, train_size=0.5, shuffle=False)

        logging.info(f'Tamanho dos conjuntos de dados: Treinamento: {len(y_train)}, Validação: {len(y_val)}, Teste: {len(y_test)}.')

        feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
        logging.info(f'Dados escalonados para o intervalo: {feature_range}.')

        arma_model = ARMA_Model(y_train, y_val)
        arma_model, arma_order, arma_val_mse = arma_model.grid_search_arma()

        train_predictions = arma_model.predict(len(y_train))
        val_predictions = arma_model.predict(len(y_train) + len(y_val))[-len(y_val):]
        test_predictions = arma_model.predict(len(y_train) + len(y_val) + len(y_test))[-len(y_test):]

        pd.DataFrame(residuals, columns=['residuals']).to_csv('residuos_arma.csv', index=False)
        logging.info('Resíduos do modelo ARMA salvos em "residuos_arma.csv".')

    except Exception as e:
        logging.error(f'Erro no processo de modelagem: {e}')
