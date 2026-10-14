#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py
pip install optuna

export PYTHONPATH=$PWD/src:$PYTHONPATH
python hyperoptimize.py $1 config/noise_schedules/$4 config/model_templates/$2 -e 20 -b 5 -o $3
