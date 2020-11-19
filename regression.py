import requests
import pandas
import scipy
import numpy
import sys
from scipy import stats


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    df = pandas.read_csv(TRAIN_DATA_URL,header=None)
    ds= df.T
    ds.columns = ['area','price']
    ds.drop([0],inplace=True)
    x_var = ds['area'].values.tolist()
    y_var = ds['price'].values.tolist()
    slope, intercept = scipy.stats.linregress(x_var, y_var)
    area = numpy.array(area)
    price = area * slope + intercept
    price = price.tolist()
    return price

if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
