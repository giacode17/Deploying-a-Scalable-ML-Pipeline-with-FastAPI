# Deploying a Scalable ML Pipeline with FastAPI

This project trains a machine learning model on the UCI Census Income dataset and serves predictions through a RESTful API built with FastAPI.

## Environment Setup

Create a conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ml
```

Alternatively, use pip:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   └── census.csv              # UCI Census Income dataset
├── ml/
│   ├── data.py                 # Data processing (one-hot encoding, label binarization)
│   └── model.py                # Model training, inference, metrics, and serialization
├── model/
│   ├── model.pkl               # Trained GradientBoostingClassifier
│   └── encoder.pkl             # Fitted OneHotEncoder
├── screenshots/
│   └── local_api.png           # Screenshot of local API responses
├── train_model.py              # Script to train and evaluate the model
├── main.py                     # FastAPI application
├── local_api.py                # Script to test API with GET and POST requests
├── test_ml.py                  # Unit tests for ML functions
├── model_card_template.md      # Model card with performance metrics
├── slice_output.txt            # Per-slice model performance on categorical features
├── environment.yml             # Conda environment specification
└── requirements.txt            # Pip requirements
```

## Training the Model

Run the training script to train the model, save it, and generate slice performance metrics:

```bash
python train_model.py
```

This will:
- Load and split the census data (80/20 train-test split)
- Train a `GradientBoostingClassifier`
- Save the model and encoder to `model/`
- Print overall precision, recall, and F1 on the test set
- Write per-slice metrics to `slice_output.txt`

## Running Tests

```bash
pytest test_ml.py -v
```

The tests cover:
- Model type validation (GradientBoostingClassifier)
- Metric computation correctness
- Inference return type and shape
- Train-test split sizes
- Data processing output types

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API exposes two endpoints:

- **GET** `/` — Returns a welcome message
- **POST** `/data/` — Accepts census features as JSON and returns an income prediction (`>50K` or `<=50K`)

### Testing the API locally

With the server running, execute:

```bash
python local_api.py
```

This sends a GET request to the root and a POST request with sample census data, printing the status codes and responses.

### Example POST request body

```json
{
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
```

## Model Card

See [model_card_template.md](model_card_template.md) for full model documentation including performance metrics, ethical considerations, and caveats.
