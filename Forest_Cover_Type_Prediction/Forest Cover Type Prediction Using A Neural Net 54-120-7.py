import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import logging

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from keras import models, layers
from keras.utils import np_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(123)  # for reproducibility


def main():
    train_path = Path.cwd()

    (x_train, y_train, x_cv, y_cv, x_test, test_id) = data_setup(train_path)

    logger.info("Data split successfully, scaling now")
    scaler = StandardScaler()
    scaler_train =scaler.fit(x_train) ## scale data
    x_train_scaled = scaler_train.transform(x_train)
    x_cv_scaled = scaler_train.transform(x_cv)
    x_test_scaled = scaler_train.transform(x_test)

    logger.info("Data scaled successfully, computing the model")
    model = fit_model(x_train_scaled, y_train)

    logger.info("Model fit successfully, running cross-validation")
    loss_and_metrics = model.evaluate(x_cv_scaled, y_cv, batch_size=64)
    print("Loss:", loss_and_metrics[0])
    print("Acc:", loss_and_metrics[1])

    logger.info("Predicting test data")
    prediction = model.predict(x_test_scaled)
    pd.DataFrame(prediction).to_csv("prediction_before.csv",
                                         index=False)
    max_tree = prediction.argmax(axis=1)
    max_tree = np.add(max_tree, 1) ## because it starts at 0
    test_prediction = pd.DataFrame({"Id": test_id,
                                    'Cover_Type': max_tree})

    logger.info("Writing result to prediction.csv")
    pd.DataFrame(test_prediction).to_csv("prediction.csv",
                                         index=False)


def extract_csv_data(train_path):
    """Returns a dict with the imported csv objects from train_path."""
    data = {}
    pattern = re.compile(".+\.csv$")
    try:
        for path in train_path.iterdir():
            if(pattern.match(path.parts[-1])):
                with path.open() as df:
                    data[path.parts[-1]]=pd.read_csv(df)
            if (not data):
                raise FileNotFoundError("No .csv file found.")
        return(data)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)


class DataSplitter:
    def __init__(self, data, fraction):
        self.splitData = self._split_train_cv_set(data, fraction)

    def train_set(self):
        return(self.splitData["train"])

    def cross_validation_set(self):
        return(self.splitData["cross-validation"])

    def _split_train_cv_set(self, data, fraction):
        """Returns a dict with data split into train and cross-validation
        set.
        """
        self.n = data.shape[0]
        self.train_n = int(round(fraction*self.n))
        self.sample = np.random.choice(np.arange(0, self.n), self.train_n, replace=False)
        self.train = data.iloc[self.sample,:]
        self.cv = data.drop(self.sample)
        return({'train':self.train, 'cross-validation':self.cv})


def data_setup(train_path):
    """Sets up data for forest type prediction. Expects train.csv and test.csv
    in train_path"""
    data = extract_csv_data(train_path)

    try:
        train_cv = data["train.csv"]
        test = data["test.csv"]
    except KeyError as e:
        logger.error("Provide a %s", str(e))
        sys.exit(1)

    split = DataSplitter(train_cv, fraction=0.8)
    train = split.train_set()
    cv = split.cross_validation_set()

    x_train = train.drop(["Id", "Cover_Type"], axis=1).values
    y_train_temp = train.loc[:,["Cover_Type"]].values
    y_train_temp = np.subtract(y_train_temp, 1) ## so that we get 7 categories instead of 8
    y_train = np_utils.to_categorical(y_train_temp)

    x_cv = cv.drop(["Id", "Cover_Type"], axis=1).values
    y_cv_temp = cv.loc[:,["Cover_Type"]].values
    y_cv_temp = np.subtract(y_cv_temp, 1)
    y_cv = np_utils.to_categorical(y_cv_temp)

    x_test = test.drop(["Id"], axis=1).values
    test_id = test["Id"].values

    return(x_train, y_train, x_cv, y_cv, x_test, test_id)


def fit_model(x_train, y_train):
    """ Fits a fully connected neural network to the input data using one
    hidden layer with 120 neurons. For the prediction it uses a softmax layer
    which normalises predictions so they can be interpreted as class
    probabilities.
    """
    model = models.Sequential()

    model.add(layers.Dense(units=120, input_dim=54))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(units=7))
    model.add(layers.Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    try:
        model.fit(x_train, y_train,
                  epochs=100,
                  batch_size=32)
    except RuntimeError:
        logger.error("Model was never compiled: %s", str(e))
        raise
    except ValueError:
        logger.error("Mismatch of input and model expectation: %s", str(e))
        raise

    return(model)


if __name__ == "__main__":
    main()
