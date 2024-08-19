import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import kerastuner as kt


# Configuração do log
logging.basicConfig(
    filename='LOG-MODELLING.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

class LSTM_Model:
    def __init__(self):
        self.tuner = None

    def build_lstm_model(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units1', min_value=20, max_value=100, step=20), return_sequences=True,
                       input_shape=(None, 1)))
        model.add(LSTM(units=hp.Int('units2', min_value=20, max_value=100, step=20)))
        model.add(Dense(1))

        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']), loss='mean_squared_error')
        return model

    def tune_lstm(self, X_train, y_train, X_val, y_val):
        self.tuner = kt.Hyperband(
            self.build_lstm_model,
            objective='val_loss',
            max_epochs=20,
            factor=3,
            directory='my_dir',
            project_name='lstm_tuning'
        )

        self.tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        model = self.build_lstm_model(best_hps)
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_data=(X_val, y_val))

        return model

