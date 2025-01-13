# !/bin/bash

# Usage: bash scripts/data.sh

python src/process_robocasa_files.py \
    "/mnt/disk_1/guanxing/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/" \
    "datasets/robocasa"
