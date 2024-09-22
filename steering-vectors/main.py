from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
from typing import List
import logging
from hooks import add_forward_hooks, get_extract_mean_activation_hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def calculate_direction(activations1: List[torch.Tensor], activations2: List[torch.Tensor]):
  """Calculate the control direction between two sets of activations by taking the mean of 
  the differences between the two sets of activations.

  Args:
      activations1 (List[torch.Tensor]): Activations of the first set of examples. Should be a list of tensors of shape (model_dim*num_layers).
      activations2 (List[torch.Tensor]): Activations of the second set of examples. Should be a list of tensors of shape (model_dim*num_layers).

  Returns:
      torch.Tensor: The mean of the differences between the two sets of activations for each residual stream dimension of each layer. 
      Shape is (1, model_dim*num_layers).
  """
  activations1 = torch.stack(activations1)
  activations2 = torch.stack(activations2)
  diff = activations2 - activations1
  return diff.mean(dim=0).unsqueeze(0)

def get_activations(model, tokenizer, examples: List[str]):
  """Get the activations of the model at each dimension of the residual stream of each layer for a list of examples.

  Args:
      model (HuggingFace-model): The LLM used in the experiments for text generation.
      tokenizer (HuggingFace-tokenizer): The tokenizer of the LLM.
      examples (List[str]): A list of example prompts for which the activations should be calculated.

  Returns:
      List[torch.Tensor]: A list of tensors of shape (model_dim*num_layers) containing the activations of the model for each example.
  """
  all_activations = []
  for i, example in enumerate(examples):
    logging.info(f"Processing example: {i}")
    activations = []
    module_hooks = [(layer, get_extract_mean_activation_hook(activations)) for layer in model.transformer.h] # Bloom architecture stores layers behind transformer.h

    with torch.no_grad(), add_forward_hooks(module_hooks):
      input_ids = tokenizer(example, return_tensors="pt").input_ids.to(device)
      _ = model(input_ids)

    activations_cached = [act.to("cuda:0") for act in activations]
    activations_all_layers = torch.cat(activations_cached, dim=0) # (num_layers * model_dim)
    all_activations.append(activations_all_layers)
  return all_activations

def get_control_direction(model, tokenizer, examples1: List[str], examples2: List[str]):
  """Calculates the control direction of the residual stream using contrastive examples.

  Args:
      model (HuggingFace-model): The LLM used in the experiments for text generation.
      tokenizer (HuggingFace-tokenizer): The tokenizer of the LLM
      examples1 (List[str]): List of examples used for calculating the control direction.
      examples2 (List[str]): The contrastive list of examples used for calculating the control direction.

  Returns:
      torch.Tensor: The residual stream control direction for all layers. Shape is (1, model_dim*num_layers).
  """
  logging.info("Extracting activations for examples1")
  activations1 = get_activations(model, tokenizer, examples1)
  logging.info("Extracting activations for examples2")
  activations2 = get_activations(model, tokenizer, examples2)
  logging.info("Calculating control direction")
  control_direction = calculate_direction(activations1, activations2)
  return control_direction, activations1, activations2

def load_sequences(filepath: str):
    """Load a list of sequences from a file.

    Args:
        filepath (str): The path to the file containing the sequences.

    Returns:
        List[str]: A list of sequences.
    """
    with open(filepath, "r") as f:
        sequences = f.readlines()
    return sequences

def main(experiment_name: str, model_name: str, filepath1: str, filepath2: str, num_examples: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Loading and quantizing the model to 4 bits")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) # Quantize to 4bit to save resources
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map = "auto"
        )
    model.eval()

    logging.info(f"Loading examples from file: {filepath1}")
    sentences1 = load_sequences(filepath1)
    sentences1 = sentences1[:num_examples]
    logging.info(f"Loading examples from file: {filepath2}")
    sentences2 = load_sequences(filepath2)
    sentences2 = sentences2[:num_examples]

    control_direction, activations1, activations2 = get_control_direction(model, tokenizer, sentences1, sentences2)
    activations1_stacked = torch.stack(activations1) # (num_examples, model_dim)
    activations2_stacked = torch.stack(activations2) # (num_examples, model_dim)

    logging.info("Saving the control direction and activations to disk")
    torch.save(control_direction, f"control_direction_mean_{experiment_name}.pt")
    torch.save(activations1_stacked, f"activations1_mean_{experiment_name}.pt")
    torch.save(activations2_stacked, f"activations2_mean_{experiment_name}.pt")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise ValueError("No GPU available.")
    experiment_name = sys.argv[1]
    model_name = sys.argv[2]
    filepath1 = sys.argv[3]
    filepath2 = sys.argv[4]
    num_examples = int(sys.argv[5])
    main(experiment_name, model_name, filepath1, filepath2, num_examples)