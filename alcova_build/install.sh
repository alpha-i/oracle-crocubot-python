#!/usr/bin/env bash

CONDA_ENV=$1

conda create -n $CONDA_ENV python=3.5 numpy=1.14.0

source activate $CONDA_ENV

pip install -r requirements.txt
pip install tensorflow-gpu

python setup.py develop


