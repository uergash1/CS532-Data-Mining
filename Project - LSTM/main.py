import numpy as np
import pandas as pd
from data_processing import load_overall_data, data_preprocessing, load_key_products_data, get_key_product_IDs, save_predictions
from lstm_driver import run_lstm

# fix random seed for reproducibility
np.random.seed(7)

# configuration for LSTM
conf = {
    "BATCH_SIZE" : 1,
    "EPOCHS_IN_RANGE" : 20,
    "EPOCHS_IN_MODEL" : 1,
    "L1_NEURONS" : 10,
    "L2_NEURONS" : 6,
    "DROPOUT" : 0.6,
    "LOSS_FUNCTION" : 'mean_squared_error',
    "OPTIMIZER" : 'adam',
    "VERBOSE" : 2
}

def main():
    file_path = 'data/product_distribution_training_set.txt'
    
    # prediction for the overall sale quantity of the 100 key products for each day 
    series = load_overall_data(file_path)
    scaler, data_scaled, raw_values = data_preprocessing(series)
    predictions = run_lstm(conf, data_scaled, scaler, raw_values)
    predictions.insert(0, 0)
    save_predictions(predictions)

    # get key product IDs
    key_product_IDs_path = 'data/key_product_IDs.txt'
    key_product_IDs = get_key_product_IDs(key_product_IDs_path)

    # prediction for the overall sale quantity for each key product for each day
    dataset = load_key_products_data(file_path)
    for key_product_ID in key_product_IDs:
        print("======== Key product ID:", key_product_ID[0], "============")
        series = dataset.loc[key_product_ID[0]]
        scaler, data_scaled, raw_values = data_preprocessing(series)
        predictions = run_lstm(conf, data_scaled, scaler, raw_values)
        predictions.insert(0, key_product_ID[0])
        save_predictions(predictions)

if __name__ == '__main__':
  main()