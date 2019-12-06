# False structure framework version 2.0

This framework is for fast prototyping of false structure examples.

## Overview
To train a network run the `Demo_start_training_proceedure.py` file. This file
will read the config.yml file and train a model (network) based on the settings
in this file.

To choose your own network architecture, implement it as a keras model in
model_builders.py

To create your own dataset, make a subclass of the abstract Data_loader class
(located in `Data_loader.py`).  You will have to implement a function called
`load_data()`.
