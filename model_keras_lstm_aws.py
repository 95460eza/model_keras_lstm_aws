# Import pytest for unit testing
import numpy
import pytest
pytest

# https://towardsdatascience.com/how-to-build-and-deploy-a-machine-learning-model-with-fastapi-64c505213857
# Create an instance of an API object
import fastapi
from fastapi import FastAPI
app_lstm = FastAPI()

# Wrap FastAPI ASGI code into a Mangum adapter (interpreter)
import mangum
from mangum import Mangum
handler = Mangum(app_lstm)
#handler2 = Mangum(model_keras_lstm)


@app_lstm.get(
    "/"
)  # The INDEX page of the CLIENT EndPoint will return a SIMPLE JSON file Object
def index():
    return {"message": "Hello, stranger"}


@app_lstm.get("/{name}")
def get_name(name: str):
    return {"message": f"Hello, {name}"}


# You can enter  http://127.0.0.1:8000/Mark and find on that website the result "Hello MarK"


# Declare the predict EndPoint on the server (access via JSON request ("post") ). Here it is http://127.0.0.1:8000/predict
@app_lstm.post("/predict")
# Function that defines how to calculate predictions from the JSON input received (here a tweet features in format for LSTM)
def predict_sentiments_probability(data_as_json):
    # Make the input JSON into a dictionary
    data = json.loads(data_as_json)

    # Make the dictionary into a dataframe
    df = pd.DataFrame.from_dict(data)

    # Model's probability prediction
    probability = model_keras_lstm.predict(df)

    # https://stackoverflow.com/questions/74005747/valueerror-typeerrornumpy-int32-object-is-not-iterable-typeerrorvars
    return {"probability": probability[0][0].item()}

