""" Predicting Forest Cover Type
This python script can be run from the command line and expects to be in the
folder that contains the data. It produces a file called prediction.csv with
the predictions of the forest cover type. It does that by first training a
neural network with one hidden layer on a training set with 80% of the
observations (20% were reserved for the cross-validation set) and then using
this network to make predictions on the test data set. The network architecure
was copied from Blackard, Jock and Denis (2000), the original publication this
dataset was used in. The model is trained using Keras and the TensorFlow
backend. Defaults for the fit procedure are hard coded.

The prediction accuracy achieved is 70%, thereby matching the accuracy achieved
by Blackard et al. The prediction accuracy of the CV set was at around 84%,
which was very close to the training accuracy. This allows the conclusion that
the test set accuracy could be improved (through feature engineering).
"""

import re
import sys
import logging

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras import models, layers
from keras.utils import np_utils

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

np.random.seed(123)  # for reproducibility

def main():
    """Extracts the data, trains a neural network and writes the predictions
    into predictions.csv.
    """

    (x_train, y_train, x_cv, y_cv, x_test, test_id) = data_setup(Path.cwd())

    LOGGER.info("Data split successfully, scaling now")
    scaler_train = StandardScaler().fit(x_train) ## scale data
    x_train_scaled = scaler_train.transform(x_train)
    x_cv_scaled = scaler_train.transform(x_cv)
    x_test_scaled = scaler_train.transform(x_test)

    LOGGER.info("Data scaled successfully, computing the model")
    model = fit_model(x_train_scaled, y_train)

    LOGGER.info("Model fit successfully, running cross-validation")
    loss_and_metrics = model.evaluate(x_cv_scaled, y_cv, batch_size=64)
    print("Loss:", loss_and_metrics[0])
    print("Acc:", loss_and_metrics[1])

    LOGGER.info("Predicting test data")
    prediction = model.predict(x_test_scaled)
    max_tree = prediction.argmax(axis=1)
    max_tree = np.add(max_tree, 1) ## because it starts at 0
    test_prediction = pd.DataFrame({"Id": test_id,
                                    'Cover_Type': max_tree})

    LOGGER.info("Writing result to prediction.csv")
    pd.DataFrame(test_prediction).to_csv("prediction.csv",
                                         index=False)


def extract_csv_data(train_path):
    """Returns a dict with the imported csv objects from train_path."""
    data = {}
    pattern = re.compile(r".+\.csv$")
    try:
        for path in train_path.iterdir():
            if pattern.match(path.parts[-1]):
                with path.open() as csv_file:
                    data[path.parts[-1]] = pd.read_csv(csv_file)
            if not data:
                raise FileNotFoundError("No .csv file found.")
        return data
    except FileNotFoundError as error:
        LOGGER.error(error)
        sys.exit(1)


class DataSplitter:
    """ Splits a data set into training and cross-validation set at the
    specified fraction.

    Provides method train_set to return a subset for training a model and
    method cross_validation_set to return a subset for cross-validating a
    model.
    """
    def __init__(self, data, fraction):
        self.split_data = self._split_train_cv_set(data, fraction)

    def train_set(self):
        """ Returns training set."""
        return self.split_data["train"]

    def cross_validation_set(self):
        """ Returns a cross-validation set."""
        return self.split_data["cross-validation"]

    def _split_train_cv_set(self, data, fraction):
        """Returns a dict with data split into train and cross-validation set."""
        self.sample_size = data.shape[0]
        self.train_n = int(round(fraction*data.shape[0]))
        self.sample = np.random.choice(np.arange(0, self.sample_size),
                                       self.train_n,
                                       replace=False)
        return{'train': data.iloc[self.sample, :],
               'cross-validation': data.drop(self.sample)}


def data_setup(train_path):
    """Sets up data for forest type prediction. Expects train.csv and test.csv
    in train_path"""
    data = extract_csv_data(train_path)

    try:
        train_cv = data["train.csv"]
        test = data["test.csv"]
    except KeyError as error:
        LOGGER.error("Provide a %s", str(error))
        sys.exit(1)

    split = DataSplitter(train_cv, fraction=0.8)
    train = split.train_set()
    cross_validation = split.cross_validation_set()

    x_train = train.drop(["Id", "Cover_Type"], axis=1).values
    y_train = train.loc[:, ["Cover_Type"]].values
    y_train = np.subtract(y_train, 1) ## so that we get 7 categories instead of 8
    y_train = np_utils.to_categorical(y_train)

    x_cv = cross_validation.drop(["Id", "Cover_Type"], axis=1).values
    y_cv = cross_validation.loc[:, ["Cover_Type"]].values
    y_cv = np.subtract(y_cv, 1)
    y_cv = np_utils.to_categorical(y_cv)

    x_test = test.drop(["Id"], axis=1).values
    test_id = test["Id"].values

    return x_train, y_train, x_cv, y_cv, x_test, test_id


def fit_model(x_train, y_train):
    """Fits a fully connected neural network to the input data using one hidden
    layer with 120 neurons. For the prediction it uses a softmax layer which
    normalises predictions so they can be interpreted as class probabilities.
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
    except RuntimeError as error:
        LOGGER.error("Model was never compiled: %s", str(error))
        raise
    except ValueError as error:
        LOGGER.error("Mismatch of input and model expectation: %s", str(error))
        raise

    return model


if __name__ == "__main__":
    main()
