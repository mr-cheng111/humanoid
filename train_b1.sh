#!/bin/bash

# B1 Humanoid Robot Training Script
# Optimized for RTX 4090 (24GB VRAM)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Change to the correct directory
cd humanoid

# Training parameters optimized for RTX 4090
TASK="b1_ppo"
RUN_NAME="b1_v1"
NUM_ENVS=4096  # Reduced from 4096 for single RTX 4090
SIM_DEVICE="cuda:0"
RL_DEVICE="0"

echo "Starting B1 training with the following parameters:"
echo "Task: $TASK"
echo "Run Name: $RUN_NAME"
echo "Number of Environments: $NUM_ENVS"
echo "Simulation Device: $SIM_DEVICE"
echo "RL Device: $RL_DEVICE"
echo "=================================="

# Start training
python scripts/train.py \
    --task=$TASK \
    --run_name=$RUN_NAME \
    --headless \
    --num_envs=$NUM_ENVS \
    --sim_device=$SIM_DEVICE \
    --rl_device=$RL_DEVICE

echo "Training completed!"
echo "To evaluate the trained policy, run:"
echo "python scripts/play.py --task=$TASK --run_name=$RUN_NAME"