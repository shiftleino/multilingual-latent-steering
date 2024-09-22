#!/bin/bash

source .venv/bin/activate

GENERATION_DIR_PATH=""
RESULT_FOLDER_PATH=""
FLUENCY_PROMPT_PATH=""
CORRECTNESS_PROMPT_PATH=""
MODEL_NAME="mistral-large-2407"
LANGUAGE="en" # fi or en
FINNISH_QUESTIONS_PATH="./finnish_questions_full.json"

# Run the script to evaluate the generated answers
pip install -r requirements.txt
python main.py $GENERATION_DIR_PATH $RESULT_FOLDER_PATH $FLUENCY_PROMPT_PATH $CORRECTNESS_PROMPT_PATH $MODEL_NAME $LANGUAGE $FINNISH_QUESTIONS_PATH

if [ $? -ne 0 ]; then
    echo "Failed to run the controlled text generation script."
    exit 1
fi
