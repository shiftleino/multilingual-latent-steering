#!/bin/bash

EXPERIMENT_NAMES=""
EXPERIMENT_TYPE="" # 'initial' or 'full'
SYSTEM_PROMTING=0 # 1 if system prompting is used, 0 otherwise
MODEL_PATH=""
STEERING_VECTOR_FILENAME=""
CONTROL_LAYERS_FILENAMES="" # should include the layers where control is applied in the form of (num_experiments, num_controlled_layers)
CONTROL_STRENGTHS=""

# Make sure that HF_HUB_OFFLINE is set to 0
export HF_HUB_OFFLINE=0

# Run the scipt for creating the control layer files
python init_control_layers.py $CONTROL_LAYERS_FILENAMES

# Run the script to generate the answers to TruthfulQA questions
python main.py $EXPERIMENT_NAMES $EXPERIMENT_TYPE $SYSTEM_PROMTING $MODEL_PATH $STEERING_VECTOR_FILENAME $CONTROL_LAYERS_FILENAMES $CONTROL_STRENGTHS

if [ $? -ne 0 ]; then
    echo "Failed to run the controlled text generation script."
    exit 1
fi
