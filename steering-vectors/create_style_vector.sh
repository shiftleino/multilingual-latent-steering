#!/bin/bash

EXPERIMENT_NAME="style-control"
MODELPATH="./model"
FILEPATH="./data/formal_sequences.txt"
NUM_EXAMPLES=157

# Run the script to create the control direction
pip install -r requirements.txt
python main.py $EXPERIMENT_NAME $MODELPATH $FILEPATH $NUM_EXAMPLES

if [ $? -ne 0 ]; then
    echo "Failed to create the control direction."
    exit 1
fi