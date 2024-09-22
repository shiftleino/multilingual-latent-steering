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

def main(generations_dir_path: str, result_folder_path: str, fluency_prompt_path: str, correctness_prompt_path: str, model_name: str, api_key: str, language: str, finnish_questions_path: str):
    logging.info("Starting the evaluation of the model generations.")
    client = Mistral(api_key=api_key)
    
    files = os.listdir(generations_dir_path)
    for file in files:
        logging.info(f"Evaluating the generations in the file: {file}")
        file_suffix = file.split("layers-")[1]
        with open(f"{generations_dir_path}/{file}") as f:
            file_content = f.read()
        generations = file_content.split("<|GENERATION_END|>")[:-1] # last element is empty due to split
        answers = [generation.split("assistant", 1)[1].strip() for generation in generations]
        results = evaluate_model_generations(answers, fluency_prompt_path, correctness_prompt_path, model_name, language, finnish_questions_path, client)
        layerset_result = {"filename": file, "results": results}
    
        with open(f"{result_folder_path}/results_{file_suffix}", "w") as json_file:
            json.dump({"results": layerset_result}, json_file, indent=4)

if __name__ == "__main__":
    generation_dir_path = sys.argv[1]
    result_folder_path = sys.argv[2]
    finnish_prompt_path = sys.argv[3]
    correctness_prompt_path = sys.argv[4]
    model_name = sys.argv[5]
    language = sys.argv[6]
    finnish_questions_path = sys.argv[7]

    load_dotenv()
    api_key = os.getenv("MISTRAL_AI_API_KEY")

    main(generation_dir_path, result_folder_path, finnish_prompt_path, correctness_prompt_path, model_name, api_key, language, finnish_questions_path)
