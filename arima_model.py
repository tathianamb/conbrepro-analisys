import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd



# Configuração do log
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


class ARIMA_Model:
    def __init__(self, df_train, df_val, df_test):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.order = None
        logging.info('ARIMA_Model instanciado.')


    def grid_search(self):
        gbest_mse = np.inf
        gbest_order = None

        for p in range(1, 5):
            for q in range(1, 5):
                try:
                    model = ARIMA(self.df_train, order=(p, 1, q))
                    model_fitted = model.fit()

                    predictions = model_fitted.predict(start=self.df_val.index[0], end=self.df_val.index[-1])

                    mse = mean_squared_error(self.df_val, predictions)

                    logging.info(f'ARIMA(p={p}, d=1, q={q}) - MSE: {mse:.4f}')

                    if mse < gbest_mse:
                        gbest_mse = mse
                        gbest_order = (p, 1, q)
                except Exception as e:
                    logging.error(f"Erro ao ajustar o modelo ARIMA(p={p}, d=1, q={q}): {e}", exc_info=True)

        self.order = gbest_order
        logging.info(f'Melhor modelo encontrado: ARIMA({self.order}) - MSE: {gbest_mse:.4f}')


    def fit_predict(self):
        model = ARIMA(pd.concat([self.df_train, self.df_val]), order=self.order)
        model_fitted = model.fit()
        self.df_train_predicted = model_fitted.predict(start=self.df_train.index[0], end=self.df_train.index[-1])
        self.df_val_predicted = model_fitted.predict(start=self.df_val.index[0], end=self.df_val.index[-1])
        self.df_test_predicted = model_fitted.predict(start=self.df_test.index[0], end=self.df_test.index[-1])
