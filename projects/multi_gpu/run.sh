#!/bin/bash
#BSUB -q long
#BSUB -n 4                           # Number of cores
#BSUB -o ./lsf_log/gpu_job_%J.out              # Output file name (%J is the job ID)
#BSUB -e ./lsf_log/gpu_job_%J.err              # Error file name
#BSUB -gpu num=3:gmem=80000:mode=shared
#BSUB -R "rusage[mem=64GB]"          # Memory requirement (specify amount needed)
#BSUB -W 24:00                        # Wall clock limit (hours:minutes)

# Define the variables
export OUTPUT_DIR="part_2"    # Change this value as needed
export DEVICES=3              # Change this value as needed
export WORKERS_PER_DEVICE=6   # Change this value as needed
export PORT=8088             # Change this value as needed
# Calculate N_JOBS
export N_JOBS=$((DEVICES * WORKERS_PER_DEVICE))

source /home/zhengx46/.bashrc
conda activate parser

# Start server in background
python server_mps_threads.py --output-dir ${OUTPUT_DIR} --devices ${DEVICES} --workers-per-device ${WORKERS_PER_DEVICE} --port ${PORT} &

# Wait a few seconds to ensure server is up
sleep 300
nvidia-smi

# Run client
python client.py --prefix ${OUTPUT_DIR} --n-jobs ${N_JOBS} --port ${PORT}


echo "==========All Client Jobs are Done=========="
