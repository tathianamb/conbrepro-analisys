import pandas as pd
import logging
from statisticalanalysis import StatisticalAnalysis
from utils import DataLoader
from preprocess import Differentiator
from arima_model import ARIMA_Model
#from lstm_model import LSTM_Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


logging.basicConfig(
    filename='LOG-MODELLING.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

def log_warning(message, category, filename, lineno, file=None, line=None):
    logging.warning(f'{message} (Category: {category.__name__}, File: {filename}, Line: {lineno})')

warnings.showwarning = log_warning

# Filtrar warnings específicos
warnings.filterwarnings('always', category=ConvergenceWarning)


if __name__ == "__main__":

    folder_path = 'DATASET2023-INPUT'
    files = [
        "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_selected.csv",
        "INMET_CO_GO_A002_GOIANIA_01-01-2023_A_31-12-2023_selected.csv",
        "INMET_NE_CE_A305_FORTALEZA_01-01-2023_A_31-12-2023_selected.csv",
        "INMET_NE_PI_A312_TERESINA_01-01-2023_A_31-12-2023_selected.csv"
    ]

    for file in files:
        df = DataLoader(file, folder_path).load_data()

        try:
            logging.debug('Iniciando separação dos conjuntos de treinamento, validação e teste.')
            df_train, df_temp = train_test_split(df, train_size=0.6, shuffle=False)
            df_val, df_test = train_test_split(df_temp, train_size=0.5, shuffle=False)
            logging.info(f'Tamanho dos conjuntos de dados: Treinamento: {df_train.size}, Validação: {df_val.size}, Teste: {df_test.size}.')

            shift = 1
            logging.debug(f'Realizando diferenciação dos dados')
            differentiator = Differentiator(column_name='radiacao_global', shift=shift)
            df_train_diff = differentiator.differentiate(df_train, 'train')
            df_val_diff = differentiator.differentiate(df_val, 'val')
            df_test_diff = differentiator.differentiate(df_test, 'test')
            logging.info(f'Diferenciação com shift = {shift}')

            df_train.name = 'file-train'
            df_val.name = 'file-val'
            df_test.name = 'file-test'

            save_folder_path = f'STATISTICALANALYSIS-DIFF{shift}'
            analysis = StatisticalAnalysis(None, save_folder_path)
            analysis.process_files(df_train)

            logging.debug('Iniciando ajuste de escala com MinMaxScaler de cada conjunto de dados.')
            feature_range = (-1, 1)
            scaler = MinMaxScaler(feature_range=feature_range)
            df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
            df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns, index=df_val.index)
            df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)
            logging.info(f'Dados escalonados para o intervalo: {feature_range}.')

            logging.debug('Instanciando ARIMA model')
            arima_model = ARIMA_Model(df_train_scaled, df_val_scaled, df_test_scaled)
            logging.info('Executando GridSearch.')
            arima_model.grid_search()
            logging.info('Ajustando e realizando previsões com o modelo ARIMA.')
            arima_model.fit_predict()

            logging.info('Calculando resíduos dos conjuntos de dados.')
            df_train_resid = df_train_scaled.sub(arima_model.df_train_predicted, axis=0)
            df_val_resid = df_val_scaled.sub(arima_model.df_val_predicted, axis=0)
            df_test_resid = df_test_scaled.sub(arima_model.df_test_predicted, axis=0)

            df_train_minmax_reverted = pd.DataFrame(scaler.inverse_transform(arima_model.df_train_predicted), columns=df_train.columns,
                                             index=df_train.index)
            df_val_minmax_reverted = pd.DataFrame(scaler.inverse_transform(arima_model.df_val_predicted), columns=df_val.columns,
                                           index=df_val.index)
            df_test_minmax_reverted = pd.DataFrame(scaler.inverse_transform(arima_model.df_test_predicted), columns=df_test.columns,
                                            index=df_test.index)

            df_train_diff_reverted = differentiator.reverse_differentiation(df_train_minmax_reverted, 'train')
            df_val_diff_reverted = differentiator.reverse_differentiation(df_val_minmax_reverted, 'val')
            df_test_diff_reverted = differentiator.reverse_differentiation(df_test_minmax_reverted, 'test')

        except Exception as e:
            logging.error(f'Erro no processo de modelagem: {e}')
