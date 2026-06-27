import pandas as pd
import numpy as np

def check_output_format(predictions: pd.DataFrame, path_to_test_data: str):
    assert predictions.shape[0] == 24 * 7, f"Predictions should have 24*7={24*7} rows, but has {predictions.shape[0]} rows"
    assert predictions.shape[1] == 2, f"Predictions should have 2 columns, but has {predictions.shape[1]} columns"
    assert predictions.columns.tolist() == ['time', 'preds'], f"Predictions should have columns 'time' and 'preds', but has {predictions.columns.tolist()}"
    assert predictions['time'].dtype == 'datetime64[ns]', f"Time should be a datetime, but is {predictions['time'].dtype}"
    assert predictions['preds'].dtype == 'float64', f"Preds should be a float, but is {predictions['preds'].dtype}"
    assert not predictions.isnull().any().any(), "Predictions should not contain any null values"    
    test_data = pd.read_csv(path_to_test_data)
    test_data = test_data.astype({"time": "datetime64[ns]", "orders": "float64"})
    comparison = predictions.merge(test_data, on='time', how='inner')
    assert comparison.shape[0] == predictions.shape[0], f"Comparison should have the same number of rows as predictions, but has {comparison.shape[0]} rows"
    print(f"MSE: {np.mean((comparison['preds'] - comparison['orders']) ** 2)}")
    print(f"Your output seems correctly formatted!")

if __name__ == "__main__":
    predictions = pd.read_csv("data/test_data_mock.csv")
    predictions = predictions.rename(columns={"time": "time", "orders": "preds"}).astype({"time": "datetime64[ns]", "preds": "float64"})[["time", "preds"]]
    check_output_format(predictions, "data/test_data_mock.csv")
