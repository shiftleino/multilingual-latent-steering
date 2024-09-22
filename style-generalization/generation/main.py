import sys
import json
from typing import List
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from generate import controlled_text_generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def create_instruction_format(example):
    """Formats the instructions to follow the instruction format of Poro-34B-chat (ChatML template).

    Args:
        example (dict): Example of the TruthfulQA dataset.

    Returns:
        dict: Example with the instruction field added.
    """
    example["instruction"] = "<|im_start|>user\n" + example["question"] + "<|im_end|>\n" + "<|im_start|>assistant\n"
    return example

def download_truthfulqa(experiment_type: str, tokenizer):
    """Downloads the TruthfulQA dataset from HuggingFace, formats the instructions to follow the 
    instruction format of Poro-34B-chat (ChatML template), and tokenizes the instructions into
    input ids.

    Args:
        experiment_type (str): The experiment type, either "initial" or "full". "Initial" uses a sample of only 10 examples,
            while "full" uses 100 examples.
        tokenizer (HuggingFace-tokenizer): The tokenizer of the LLM.

    Returns:
        List[List[Int]]: List of tokenized instructions.
    """
    truthfulqa = load_dataset("truthfulqa/truthful_qa", name="generation", split="validation")
    if experiment_type == "initial":
        logging.info("Extracting 10 examples for initial study.")
        truthfulqa = truthfulqa.select(range(10))
    elif experiment_type == "full":
        logging.info("Extracting 100 examples for full study.")
        truthfulqa = truthfulqa.select(range(100))

    logging.info("Formatting the examples to follow the instruction format of Poro-34B-chat (ChatML template)")
    truthfulqa = truthfulqa.map(create_instruction_format)

    logging.info("Tokenizing the instructions.")
    truthfulqa = truthfulqa.map(lambda x: tokenizer(x["instruction"]))
    input_ids = truthfulqa["input_ids"]
    return input_ids

def get_finnish_questions(filepath: str, experiment_type: str, tokenizer):
    """Loads the Finnish questions from a file, formats the instructions to follow the
    instruction format of Poro-34B-chat (ChatML template), and tokenizes the instructions into
    input ids.

    Args:
        filepath (str): The path to the file containing the Finnish questions.
        experiment_type (str): The experiment type, either "initial" or "full". "Initial" uses a sample of only 10 examples,
            while "full" uses 100 examples.
        tokenizer (HuggingFace-tokenizer): The tokenizer of the LLM.

    Returns:
        List[List[Int]]: List of tokenized instructions.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    questions = [{"question": question} for question in data["questions"]]
    if experiment_type == "initial":
        logging.info("Extracting 10 examples for initial study.")
        questions = questions[:10]
    elif experiment_type == "full":
        logging.info("Extracting 100 examples for full study.")
        questions = questions[:100]
    
    logging.info("Formatting the examples to follow the instruction format of Poro-34B-chat (ChatML template)")
    questions = [create_instruction_format(question) for question in questions]

    logging.info("Tokenizing the instructions.")
    input_ids = [tokenizer(question["instruction"])["input_ids"] for question in questions]
    return input_ids

def main(experiment_name: str, experiment_type: str, model_path: str, steering_vector_filename: str, control_layer_files: List[str], control_strengths: List[float], language: str, finnish_questions_filepath: str):
    logging.info("Loading the control direction and the tokenizer.")
    steering_vector = torch.load(steering_vector_filename)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info(f"Loading model from {model_path} and quantizing it.")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map = "auto"
        )
    model.eval()
    
    if language == "finnish":
        logging.info("Loading Finnish questions.")
        input_ids = get_finnish_questions(finnish_questions_filepath, experiment_type, tokenizer)
    elif language == "english":
        logging.info("Loading TruthfulQA validation set from HuggingFace as the evaluation set.")
        input_ids = download_truthfulqa(experiment_type, tokenizer)
    else:
        raise ValueError("Language must be either 'finnish' or 'english'.")

    for control_strength in control_strengths:
        logging.info(f"Control strength: {control_strength}")
        for i, layer_file_name in enumerate(control_layer_files):
            logging.info(f"Loading layer information from {layer_file_name}.")
            layers = torch.load(layer_file_name)
            logging.info("Starting controlled text generation.")
            for i in range(layers.shape[0]):
                logging.info(f"Generating answers with controlled layers: {layers[i, :]}.")
                generated_tokens, logits, activations = controlled_text_generation(model, tokenizer, input_ids, steering_vector, layers[i, :], control_strength)
                logging.info("Decoding the generated tokens.")
                generated_answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
                logging.info("Saving the generated answers and the logits and activations of the last instruction token.")
                with open(f"results/{experiment_type}/generated_answers_{experiment_name}_layerset_idx{i}_{control_strength}.txt", "w") as f:
                    for answer in generated_answers:
                        f.write(answer + "\n\n<|GENERATION_END|>\n\n")
                torch.save(logits, f"results/{experiment_type}/last_instruct_token_logits_{experiment_name}_layerset_idx{i}_{control_strength}.pt")
                torch.save(activations, f"results/{experiment_type}/last_instruct_token_activations_{experiment_name}_layerset_idx{i}_{control_strength}.pt")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("No GPU available.")
    experiment_name = sys.argv[1]
    experiment_type = sys.argv[2]
    model_path = sys.argv[3]
    steering_vector_filename = sys.argv[4]
    control_layer_filenames = sys.argv[5]
    control_layer_filenames = control_layer_filenames.split(",")
    control_strengths = [float(strength) for strength in sys.argv[6].split(",")]
    language = sys.argv[7]
    finnish_questions_filepath = sys.argv[8]
    main(experiment_name, experiment_type, model_path, steering_vector_filename, control_layer_filenames, control_strengths, language, finnish_questions_filepath)
