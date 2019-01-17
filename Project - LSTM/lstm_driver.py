from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from data_processing import invert_scale_data, get_inverse_difference

# fit an LSTM network to training data
def fit_lstm(conf, data_scaled):
    X, y = data_scaled[:, 0:-1], data_scaled[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(conf["L1_NEURONS"], batch_input_shape=(conf["BATCH_SIZE"], X.shape[1], X.shape[2]), stateful=True, return_sequences=True, recurrent_dropout=conf["DROPOUT"]))
    model.add(LSTM(conf["L2_NEURONS"], batch_input_shape=(conf["BATCH_SIZE"], X.shape[1], X.shape[2]), stateful=True, recurrent_dropout=conf["DROPOUT"]))
    model.add(Dense(1))
    model.compile(loss=conf["LOSS_FUNCTION"], optimizer=conf["OPTIMIZER"])
    for i in range(conf["EPOCHS_IN_RANGE"]):
        model.fit(X, y, epochs=conf["EPOCHS_IN_MODEL"], batch_size=conf["BATCH_SIZE"], verbose=conf["VERBOSE"], shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, X, conf):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=conf["BATCH_SIZE"])
	return yhat[0,0]

def run_lstm(conf, data_scaled, scaler, raw_values):
    print("Predicting...")
    # fit the model
    lstm_model = fit_lstm(conf, data_scaled)
    # forecast the entire training dataset to build up state for forecasting
    data_reshaped = data_scaled[:, 0].reshape(len(data_scaled), 1, 1)
    lstm_model.predict(data_reshaped, batch_size=conf["BATCH_SIZE"])

    X = np.array([data_scaled[-1, -1]])

    # walk-forward validation on the test data
    predictions = list()
    for i in range(0, 29):
        yhat = forecast_lstm(lstm_model, X, conf)
        next_input = yhat
        yhat = invert_scale_data(scaler, X, yhat)
        yhat = get_inverse_difference(raw_values, yhat, len(data_scaled)+1-i)
        if type(yhat) is np.ndarray:
            if yhat[0] > 0:
                predictions.append(yhat[0])
            else:
                predictions.append(0)
        else:
            if yhat > 0:
                predictions.append(yhat)
            else:
                predictions.append(0)
        X = np.array([next_input])

    return predictions