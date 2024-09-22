#!/bin/bash

EXPERIMENT_NAME=""
EXPERIMENT_TYPE="" # 'initial' or 'full'
MODEL_PATH=""
STEERING_VECTOR_FILENAME=""
CONTROL_LAYERS_FILENAMES="" # should include the layers where control is applied in the form of (num_experiments, num_controlled_layers)
CONTROL_STRENGTHS=""
LANGUAGE="" # 'finnish' or 'english'
FINNISH_QUESTIONS_FILEPATH="./finnish_questions_full.json" # used if LANGUAGE is 'finnish'

# Make sure that HF_HUB_OFFLINE is set to 0
export HF_HUB_OFFLINE=0

# Run the scipt for creating the control layer files
python init_control_layers.py $CONTROL_LAYERS_FILENAMES

# Run the script to generate the answers to TruthfulQA questions
python main.py $EXPERIMENT_NAME $EXPERIMENT_TYPE $MODEL_PATH $STEERING_VECTOR_FILENAME $CONTROL_LAYERS_FILENAMES $CONTROL_STRENGTHS $LANGUAGE $FINNISH_QUESTIONS_FILEPATH

if [ $? -ne 0 ]; then
    echo "Failed to run the controlled text generation script."
    exit 1
fi
