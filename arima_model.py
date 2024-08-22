import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class ARIMA_Model:
    def __init__(self, df_train, df_val, df_test, feature_range = (0, 1)):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.df_train_scaled = None
        self.df_val_scaled = None
        self.df_test_scaled = None

        self.df_train_predicted = None
        self.df_val_predicted = None
        self.df_test_predicted = None

        self.df_train_unscaled = None
        self.df_val_unscaled = None
        self.df_test_unscaled = None

        self.order = None
        self.scaler = MinMaxScaler(feature_range=feature_range)
        logging.info('ARIMA_Model instanciado.')

    def scale_data(self):
        feature_range = (-1, 1)
        self.df_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.df_train), columns=self.df_train.columns, index=self.df_train.index)
        self.df_val_scaled = pd.DataFrame(self.scaler.transform(self.df_val), columns=self.df_val.columns, index=self.df_val.index)
        self.df_test_scaled = pd.DataFrame(self.scaler.transform(self.df_test), columns=self.df_test.columns, index=self.df_test.index)
        logging.info(f'Dados escalonados para o intervalo: {feature_range}.')

    def grid_search(self, d=0):
        gbest_mse = np.inf
        gbest_order = None

        for p in range(1, 5):
            for q in range(1, 5):
                try:
                    model = ARIMA(self.df_train, order=(p, d, q))
                    model_fitted = model.fit()

                    predictions = model_fitted.predict(start=self.df_val.index[0], end=self.df_val.index[-1])

                    mse = mean_squared_error(self.df_val, predictions)

                    logging.info(f'ARIMA(p={p}, d=1, q={q}) - MSE: {mse:.4f}')

                    if mse < gbest_mse:
                        gbest_mse = mse
                        gbest_order = (p, 1, q)
                except Exception as e:
                    logging.error(f"Erro ao ajustar o modelo ARIMA(p={p}, d={d}, q={q}): {e}", exc_info=True)

        self.order = gbest_order
        logging.info(f'Melhor modelo encontrado: ARIMA({self.order}) - MSE: {gbest_mse:.4f}')


    def fit_predict(self):
        model = ARIMA(pd.concat([self.df_train, self.df_val]), order=self.order)
        model_fitted = model.fit()
        self.df_train_predicted = model_fitted.predict(start=self.df_train.index[0], end=self.df_train.index[-1])
        self.df_val_predicted = model_fitted.predict(start=self.df_val.index[0], end=self.df_val.index[-1])
        self.df_test_predicted = model_fitted.predict(start=self.df_test.index[0], end=self.df_test.index[-1])

    def undo_scale_data(self):
        if not hasattr(self, 'scaler'):
            raise AttributeError("Scaler has not been initialized. Please scale the data first.")

        if self.scaler is None:
            raise ValueError(
                "Scaler is not set. Please ensure that scaling was performed before attempting to undo it.")

        # Apply inverse transform to revert scaling
        self.df_train_unscaled = pd.DataFrame(self.scaler.inverse_transform(self.df_train_predicted),
                                              columns=self.df_train.columns,
                                              index=self.df_train.index)
        self.df_val_unscaled = pd.DataFrame(self.scaler.inverse_transform(self.df_val_predicted),
                                            columns=self.df_val.columns,
                                            index=self.df_val.index)
        self.df_test_unscaled = pd.DataFrame(self.scaler.inverse_transform(self.df_test_predicted),
                                             columns=self.df_test.columns,
                                             index=self.df_test.index)
        logging.info(f'Dados revertidos para o intervalo original.')

    def run_pipeline(self, steps):
        """
        Execute a sequence of steps specified by the user.

        :param steps: List of steps to execute. Each step should be a callable (method or function) or a string
                      with the name of the method to call.
        """
        available_methods = {method_name: getattr(self, method_name) for method_name in dir(self)
                             if callable(getattr(self, method_name)) and not method_name.startswith('__')}

        for step in steps:
            if isinstance(step, str):
                if step not in available_methods:
                    raise ValueError(f"Method '{step}' not found in the available methods.")
                method = available_methods[step]
            elif callable(step):
                method = step
            else:
                raise ValueError("Steps should be method names (strings) or callable objects.")

            logging.info(f"Method: {method.__name__}")
            method()