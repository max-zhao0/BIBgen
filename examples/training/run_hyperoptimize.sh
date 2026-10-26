#!/bin/bash
# Arguments: data model_template out schedule (one row of experiments_hyper.txt)

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py
pip install optuna
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# The study lives in a folder named after the output config, e.g. best_equivariant_denoiser/ holding
# best_equivariant_denoiser.db and, once every trial is done, best_equivariant_denoiser.json.
# submit_hyperoptimize.sub transfers the folder in and back out, so resubmitting resumes the study.
study_dir=${3%.json}
mkdir -p "$study_dir"

export PYTHONPATH=$PWD/src:$PYTHONPATH
python hyperoptimize.py $1 config/noise_schedules/$4 config/model_templates/$2 -e 20 -b 5 -n 40 --timeout 20 -o "$study_dir/$3"
