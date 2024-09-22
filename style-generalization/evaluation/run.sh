#!/bin/bash

source .venv/bin/activate

GENERATION_FILE_PATH=""
BASELINE_FILE_PATH=""
RESULT_FOLDER_PATH=""
COMPARISON_PROMPT_PATH=""
FLUENCY_PROMPT_PATH=""
MODEL_NAME="mistral-large-2407"
LANGUAGE="" # "en" or "fi"
FINNISH_QUESTIONS_PATH="./finnish_questions_full.json"

# Run the script to evaluate the generated answers
pip install -r requirements.txt
python main.py $GENERATION_FILE_PATH $BASELINE_FILE_PATH $RESULT_FOLDER_PATH $COMPARISON_PROMPT_PATH $FLUENCY_PROMPT_PATH $MODEL_NAME $LANGUAGE $FINNISH_QUESTIONS_PATH

if [ $? -ne 0 ]; then
    echo "Failed to run the controlled text generation script."
    exit 1
fi
