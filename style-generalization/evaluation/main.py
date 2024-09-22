import os
import sys
import json
import logging
from dotenv import load_dotenv
from mistralai import Mistral
from evaluate import evaluate_model_generations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main(generations_file_path: str, baseline_file_path: str, result_folder_path: str, comparison_prompt_path: str, fluency_prompt_path: str, model_name: str, api_key: str, language: str, finnish_questions_path: str):
    logging.info("Starting the evaluation of the model generations.")
    client = Mistral(api_key=api_key)
    with open(generations_file_path, "r") as f:
        file_content = f.read()
    
    logging.info(f"Evaluating the generations in the file: {generations_file_path}")
    file_suffix = generations_file_path.split("answers_")[1]
    generations = file_content.split("<|GENERATION_END|>")[:-1] # last element is empty due to split
    answers = [generation.split("assistant", 1)[1].strip() for generation in generations]

    with open(baseline_file_path, "r") as f:
        file_content = f.read()
    generations = file_content.split("<|GENERATION_END|>")[:-1] # last element is empty due to split
    baselines = [generation.split("assistant", 1)[1].strip() for generation in generations]

    results = evaluate_model_generations(answers, baselines, comparison_prompt_path, fluency_prompt_path, model_name, client, language, finnish_questions_path)
    logging.info(f"Writing the results to {result_folder_path}/results_{file_suffix}")
    with open(f"{result_folder_path}/results_{file_suffix}", "w") as json_file:
        json.dump({"results": results}, json_file, indent=4)

if __name__ == "__main__":
    generation_file_path = sys.argv[1]
    baseline_file_path = sys.argv[2]
    result_folder_path = sys.argv[3]
    comparison_prompt_path = sys.argv[4]
    fluency_prompt_path = sys.argv[5]
    model_name = sys.argv[6]
    language = sys.argv[7]
    finnish_questions_path = sys.argv[8]

    load_dotenv()
    api_key = os.getenv("MISTRAL_AI_API_KEY")

    main(generation_file_path, baseline_file_path, result_folder_path, comparison_prompt_path, fluency_prompt_path, model_name, api_key, language, finnish_questions_path)
