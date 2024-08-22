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

            logging.debug('Instanciando ARIMA model')
            arima_model = ARIMA_Model(df_train, df_val, df_test)
            arima_model.run_pipeline(['scale_data', 'grid_search', 'fit_predict', 'undo_scale_data'])

            logging.debug(f'Realizando diferenciação dos dados')
            df_train_diff_reverted = differentiator.reverse_differentiation(arima_model.df_train_unscaled, 'train')
            df_val_diff_reverted = differentiator.reverse_differentiation(arima_model.df_val_unscaled, 'val')
            df_test_diff_reverted = differentiator.reverse_differentiation(arima_model.df_test_unscaled, 'test')
            logging.info(f'Revertendo diferenciação dos dados')

            df_train_resid = df_train.sub(df_train_diff_reverted, axis=0)
            df_val_resid = df_val.sub(df_val_diff_reverted, axis=0)
            df_test_resid = df_test.sub(df_test_diff_reverted, axis=0)
            logging.info('Resíduos calculados para cada conjunto de dados.')

        except Exception as e:
            logging.error(f'Erro no processo de modelagem: {e}')
