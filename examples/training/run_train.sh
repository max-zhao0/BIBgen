#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

export PYTHONPATH=$PWD/src:$PYTHONPATH
python train.py $1 config/noise_schedules/$4 config/models/$2 -e 151 -b 5 -t $3 --loss ${5:-simple}
