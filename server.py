"""
The Server module that is the simpler of the two entities.

Unlike the Client, which does a lot of computations, this
code has one, arguably simple task: it exposes an endpoint
that accepts post requests with serialised Keras models and
combines them with the current global model it maintains.
Then it sends the model back to the requesting Client.

We use Tornado to handle the requests as it allows us to
write simple, efficient code without many assumptions.

This module is run as a standalone app:
```
    ./fml_nets/run_server
```
"""
from typing import List

from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado.escape import json_decode
from asyncio import Lock
from datetime import datetime
from keras.models import load_model, Sequential

import pandas as pd
import numpy as np

BASE_URL = 'localhost'
PORT = 8000


class DistributedTrainerServer:
    """A class encapsulating the global model and providing
    convenience methods for manipulating its state"""
    def __init__(self):
        self.model = None
        self.lock = Lock()
        self.host_timestamps = {}
        self.no_updates = 0
        self.model_file_path = 'main_model.h5'

    def add_model_from_file(self, filename: str):
        print('[TrainerServer] combining models.')
        new_model = load_model(filename)
        # we need to provide exclusive access while combining:
        with (yield from self.lock):
            self.update_model(new_model)
            # save the model to a file for everyone to read:
            self.model.save(self.model_file_path)
            self.host_timestamps[filename] = datetime.now()
            self.no_updates += 1

    def update_model(
        self, new_model: Sequential,
    ) -> None:
        """Combines a new model with the existing one by means of weight
        averaging, assigning to the server side model weight proportional
        to the number of updates already made."""
        if not self.model:
            self.model = new_model

        # assign increased weights to the base model:
        base_weights = [w * self.no_updates for w in self.model.get_weights()]
        new_weights = new_model.get_weights()
        combined_weights = [
            (w1 + w2) / (self.no_updates * 2)
            for (w1, w2) in zip(base_weights, new_weights)
        ]
        self.model.set_weights(combined_weights)

    @property
    def current_model(self):
        with (yield from self.lock):
            return self.model

trainer_server = DistributedTrainerServer()


class ModelReceiveHandler(RequestHandler):
    """Handler for the incoming models."""
    def post(self, model_id):
        global trainer_server
        # read the data from the request:
        data = self.request.body
        model_id = json_decode(model_id)
        print('POST request ("/model/model/{}")'.format(model_id))
        filename = str(model_id) + '.h5'

        # write the model to the file:
        with open(filename, 'wb') as f:
            f.write(data)
            f.flush()

        # combine the model:
        trainer_server.add_model_from_file(filename)
        # read the global model:
        model_file = open(trainer_server.model_file_path, 'rb').read()
        self.write(model_file)
        model_file.close()


class DisplayResultsHandler(RequestHandler):
    """Handler that displays the times of update
    for each of the clients.

    For debugging and informational purposes only.
    """
    def get(self):
        print('GET request ("/")')
        self.write(
            'Last received updates by host: {}'.format(
                str(trainer_server.host_timestamps),
            )
        )


def make_app():
    return Application([
        (r'/model/([0-9]+)', ModelReceiveHandler),
        (r'/', DisplayResultsHandler),
    ])


def main():
    print('Running the server on port ', PORT)
    app = make_app()
    app.listen(PORT)
    IOLoop.current().start()


if __name__ == '__main__':
    main()
