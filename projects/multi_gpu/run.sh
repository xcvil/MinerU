#!/bin/bash
#BSUB -q long
#BSUB -n 8                           # Number of cores
#BSUB -o ./lsf_log/gpu_job_%J.out              # Output file name (%J is the job ID)
#BSUB -e ./lsf_log/gpu_job_%J.err              # Error file name
#BSUB -gpu num=2:gmem=80000:mode=shared
#BSUB -R "rusage[mem=64GB]"          # Memory requirement (specify amount needed)
#BSUB -W 24:00                        # Wall clock limit (hours:minutes)

# Define the variables
export PREFIX="part_3"    
export DEVICES=2              
export WORKERS_PER_DEVICE=8   
export PORT=6000             
# Calculate N_JOBS
export N_JOBS=$((DEVICES * WORKERS_PER_DEVICE))
export OUTPUT_DIR="/pstore/data/llm-comptox/Input/RDR_232_MM/${PREFIX}"

source /home/zhengx46/.bashrc
conda activate parser

# Start server in background
python server_mps_threads.py --output-dir ${OUTPUT_DIR} --devices ${DEVICES} --workers-per-device ${WORKERS_PER_DEVICE} --port ${PORT} &

# Wait a few seconds to ensure server is up
sleep 300
nvidia-smi

# Run client
python client.py --prefix ${PREFIX} --n-jobs ${N_JOBS} --port ${PORT}
