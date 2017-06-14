import requests
from celery import Celery
import pandas as pd
from keras.models import load_model, Sequential
from learners import OnlineLearner, MLPLearner

app = Celery('tasks', broker='redis://localhost', backend='redis://localhost')


def send_model(base_url: str, name: str):
    """Send a file to a given url"""
    url = 'http://{}:8000/model/{}'.format(base_url, name)
    with open(name, 'rb') as f:
        data = f.read()

    headers = {'Content-Type': 'application/octet-stream'}
    res = requests.post(url, data=data, headers=headers)

    with open(name, 'wb') as f:
        f.write(res.content)


@app.task
def train_model_task(learner: OnlineLearner, batch: pd.DataFrame):
    learner.train_on_batch(batch)
    return learner


@app.task
def train_and_send_task(
    name: str, url: str, learner: MLPLearner, batch: pd.DataFrame,
) -> Sequential:
    # first train the model
    print('[task] training the model')
    learner.train_on_batch(batch)
    # and then save it to file
    print('[task] serializing the model ({})'.format(name))
    learner.serialize_model(name)
    # and send it to the server
    print('[task] sending the model to the server')
    send_model(url, name)
    # load the received file into a keras model object:
    model = load_model(name)
    print('[task] done')
    return model
