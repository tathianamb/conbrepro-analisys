from arima_model import ARIMA_Model
from lstm_model import LSTM_Model
from sklearn.preprocessing import MinMaxScaler


class HybridModel:
    def __init__(self, files):
        self.files = files

    def rolling_window(self, residuals, window_size=10):
        X, y = [], []
        for i in range(len(residuals) - window_size):
            X.append(residuals[i:i + window_size])
            y.append(residuals[i + window_size])
        return np.array(X), np.array(y)

    def run(self):
        # Carregar dados
        df = DataLoader(self.files[0]).load_data()  # Usando o primeiro arquivo como exemplo
        y = df['radiacao_global'].values

        # Escalar os dados
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        train_size = int(len(y_scaled) * 0.6)
        val_size = int(len(y_scaled) * 0.2)

        y_train, y_temp = y_scaled[:train_size], y_scaled[train_size:]
        y_val, y_test = y_temp[:val_size], y_temp[val_size:]

        # Modelagem ARMA
        arma_model = ARIMA_Model(y_train, y_val)
        arma_model, arma_order, arma_val_mse, arma_val_mape = arma_model.grid_search_arma()
        residuals = arma_model.resid

        # Preparar dados para LSTM
        window_size = 10
        X, y_lstm = self.rolling_window(residuals, window_size)

        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_temp = X[:len(y_train) - window_size], X[len(y_train) - window_size:]
        y_train_lstm, y_temp_lstm = y_lstm[:len(y_train) - window_size], y_lstm[len(y_train) - window_size:]

        X_val, X_test = X_temp[:len(y_val) - window_size], X_temp[len(y_val) - window_size:]
        y_val_lstm, y_test_lstm = y_temp_lstm[:len(y_val) - window_size], y_temp_lstm[len(y_val) - window_size:]

        # Modelagem LSTM
        lstm_model = LSTM_Model()
        lstm_model = lstm_model.tune_lstm(X_train, y_train_lstm, X_val, y_val_lstm)

        # Previsões
        arma_forecast = arma_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        lstm_forecast = lstm_model.predict(X_test)

        # Combinar previsões
        combined_forecast = scaler.inverse_transform(arma_forecast.reshape(-1, 1) + lstm_forecast)

        return combined_forecast
