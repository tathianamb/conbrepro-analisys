import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import DataLoader


class TimeSeriesModel:
    def __init__(self, df_train, df_val, df_test, look_back=5, feature_range = (-1, 1)):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def preprocess_data(self):
        # Escalonar os dados
        self.df_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.df_train),
                                            columns=self.df_train.columns,
                                            index=self.df_train.index)
        self.df_val_scaled = pd.DataFrame(self.scaler.transform(self.df_val),
                                          columns=self.df_val.columns,
                                          index=self.df_val.index)
        self.df_test_scaled = pd.DataFrame(self.scaler.transform(self.df_test),
                                           columns=self.df_test.columns,
                                           index=self.df_test.index)

        # Criar os inputs
        all_X, all_y = self._create_inputs([self.df_train_scaled, self.df_val_scaled, self.df_test_scaled])
        self.train_X = np.reshape(all_X[0], (all_X[0].shape[0], 1, all_X[0].shape[1]))
        self.train_y = all_y[0]
        self.val_X = np.reshape(all_X[1], (all_X[1].shape[0], 1, all_X[1].shape[1]))
        self.val_y = all_y[1]
        self.test_X = np.reshape(all_X[2], (all_X[2].shape[0], 1, all_X[2].shape[1]))
        self.test_y = all_y[2]

    def _create_inputs(self, df_list, look_back=5):
        def add_offsets(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        all_X = []
        all_y = []
        for df in df_list:
            X, y = add_offsets(df.values, look_back)
            all_X.append(X)
            all_y.append(y)

        return all_X, all_y

    def build_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.LSTM(4, input_shape=(1, self.look_back)))
        self.model.add(keras.layers.Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self):
        if self.model is None:
            raise Exception("Model must be built before training.")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.keras", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]

        history = self.model.fit(x=self.train_X,
                                 y=self.train_y,
                                 validation_data=(self.val_X, self.val_y),
                                 epochs=100,
                                 batch_size=1,
                                 verbose=2,
                                 callbacks=callbacks)

        return history

    def evaluate_model(self):
        if self.model is None:
            raise Exception("Model must be built before evaluation.")

        test_loss = self.model.evaluate(self.test_X, self.test_y)
        return test_loss


# Uso
folder_path = 'DATASET2023-INPUT'
file = "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_selected.csv"
df = DataLoader(file, folder_path).load_data()

df_train, df_temp = train_test_split(df, train_size=0.6, shuffle=False)
df_val, df_test = train_test_split(df_temp, train_size=0.5, shuffle=False)

model = TimeSeriesModel(df_train, df_val, df_test)
model.preprocess_data()
model.build_model()
history = model.train_model()
test_loss = model.evaluate_model()

print(f"Test Loss: {test_loss}")
