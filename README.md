# Distributed approach to ML

The `fml_nets` package contains a rudimentary framework for Federated Machine Learning: it trains a Multi-Layer Perceptron on each of the devices (most likely: smartphones or IoT devices) and then sends each model to the server. The server's role is to aggregate the models by means of weight averaging and sends it back to the devices.

