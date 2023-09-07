#!/bin/bash

kaggle datasets download kaushil268/disease-prediction-using-machine-learning
unzip disease-prediction-using-machine-learning.zip -d disease-prediction
python prepare_data.py
