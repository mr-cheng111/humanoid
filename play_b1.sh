#!/bin/bash

# B1 Humanoid Robot Policy Evaluation Script
# For RTX 4090

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Change to the correct directory
cd humanoid

# Evaluation parameters
TASK="b1_ppo"
RUN_NAME="b1_v1"
SIM_DEVICE="cuda:0"
RL_DEVICE="cuda:0"

echo "Starting B1 policy evaluation with the following parameters:"
echo "Task: $TASK"
echo "Run Name: $RUN_NAME"
echo "Simulation Device: $SIM_DEVICE"
echo "RL Device: $RL_DEVICE"
echo "=================================="

# Start evaluation (this will also export JIT model for deployment)
python scripts/play.py \
    --task=$TASK \
    --run_name=$RUN_NAME \
    --sim_device=$SIM_DEVICE \
    --rl_device=$RL_DEVICE

echo "Policy evaluation completed!"
echo "JIT model exported for deployment."