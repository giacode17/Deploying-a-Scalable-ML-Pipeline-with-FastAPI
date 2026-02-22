import pytest
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

train, test = train_test_split(data, test_size=0.2, random_state=42)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)


# The first test. 
def test_model_type():
    """
    # Test that train_model returns a GradientBoostingClassifier.
    """
    assert isinstance(model, GradientBoostingClassifier)


# The second test. 
def test_compute_model_metrics():
    """
    # Test that compute_model_metrics returns expected values for known inputs.
    """
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision == 1.0
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8, abs=0.01)


# The third test. 
def test_inference_return_type():
    """
    # Test that inference returns a numpy array with the correct size.
    """
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_train_test_split_size():
    """
    # Test that the train-test split produces the expected sizes.
    """
    assert train.shape[0] == pytest.approx(data.shape[0] * 0.8, abs=1)
    assert test.shape[0] == pytest.approx(data.shape[0] * 0.2, abs=1)


def test_process_data_return_types():
    """
    # Test that process_data returns numpy arrays for X and y.
    """
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
