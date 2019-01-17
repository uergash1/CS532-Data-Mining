from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# frame a sequence as a supervised learning problem
def convert_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def get_difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def get_inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train to [-1, 1]
def scale_data(data):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(data)
	# transform train
	data = data.reshape(data.shape[0], data.shape[1])
	data_scaled = scaler.transform(data)
	return scaler, data_scaled

# inverse scaling for a forecasted value
def invert_scale_data(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# load overall (summed) sale quantity over the 100 key products for each day (1-118 days)
def load_overall_data(file_path):
    print("Loading data for overall prediction...")
    product_distribution_training_set = pd.read_csv(file_path, engine='python', sep='\t', header=None)
    #product_distribution_training_set = product_distribution_training_set.astype('float64')
    series = pd.DataFrame(data=product_distribution_training_set.sum(), columns=["quantity"])
    series = series.iloc[1:]
    return series

# load overall (summed) sale quantity of each key product for each day
def load_key_products_data(file_path):
    print("Loading data for the prediction of each key product...")
    product_distribution_training_set = pd.read_csv(file_path, engine='python', sep='\t', header=None)
    series = product_distribution_training_set.groupby(0).sum()
    return series

def get_key_product_IDs(file_path):
    key_product_IDs = pd.read_csv(file_path, engine='python', header=None)
    return key_product_IDs.values.tolist()

def data_preprocessing(series):
    print("Preprocessing the data...")
    # transform data to be stationary
    raw_values = series.values
    diff_values = get_difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = convert_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # transform the scale of the data
    scaler, data_scaled = scale_data(supervised_values)
    return scaler, data_scaled, raw_values

def save_predictions(predictions):
    print("Saving the predictions...")
    with open('output.txt', 'a+') as output_txt:
        for quantity in predictions:
            output_txt.write(str(int(quantity)) + "\t")
        output_txt.write("\n")



