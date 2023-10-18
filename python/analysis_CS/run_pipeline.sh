#!/bin/bash
conda activate sleepclass

python3 unconstrained_pipeline.py --multirun +device='[02L, 02R, 03L, 03R, 07L]' hydra/launcher=joblib