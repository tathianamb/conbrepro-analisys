import pandas as pd
import keras
from utils import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_inputs(df_list, look_back=5):
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

folder_path = 'DATASET2023-INPUT'
file = "INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023_selected.csv"

df = DataLoader(file, folder_path).load_data()

df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
#df_train, df_temp = train_test_split(df, train_size=0.6, shuffle=False)
#df_val, df_test = train_test_split(df_temp, train_size=0.5, shuffle=False)

feature_range = (-1, 1)
scaler = MinMaxScaler(feature_range=feature_range)
df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
#df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns, index=df_val.index)
df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)

#train_X, train_y, val_X, val_y, test_X, test_y = create_inputs([df_train_scaled, df_val_scaled, df_test_scaled])
all_X, all_y = create_inputs([df_train_scaled, df_test_scaled])
train_X = all_X[0]
test_X = all_X[0]
train_y = all_y[0]
test_y = all_y[1]

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
#val_X = np.reshape(val_X, (val_X.shape[0], 1, val_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

model = keras.models.Sequential()
model.add(keras.layers.LSTM(4, input_shape=(1, 1)))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x=train_X, y=train_y,validation_split=0.2,epochs=100, batch_size=1, verbose=2)

test_loss, test_acc = model.evaluate(test_X, test_y)