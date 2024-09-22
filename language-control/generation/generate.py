from typing import List
import torch
from hooks import add_forward_hooks, get_extract_last_activation_hook, get_add_control_vector_all_hook
from transformers import GenerationConfig
import logging

@torch.no_grad()
def controlled_text_generation(model, tokenizer, all_input_ids: List[List[int]], steering_vector: torch.Tensor, layers: List[int], a: float):
    """Generates text with steering vectors added to the residual stream after specified layers of the model.

    Args:
        model (HuggingFace-model): The LLM model used for the generations. Should be the same as the model used for creating
            the steering vectors.
        tokenizer (HuggingFace-tokenizer): The tokenizer of the LLM.
        all_input_ids (List[List[int]]): List of tokenized instructions.
        steering_vector (torch.Tensor): The steering vector to be added to the activations.
        layers (torch.Tensor): Tensor of layer indices where the steering vector is added.
        a (float): The scaling factor for the steering vector.

    Returns:
        tuple: A tuple containing the generated tokens, the logits of the last instruction token, 
            and the representations of the last instruction token.
    """
    generation_config = GenerationConfig(
        max_new_tokens=300,
        do_sample=True, # use greedy decoding as this is the standard decoding method for TrutfhulQA
        stop_strings=[tokenizer.eos_token]
    )
    model_dim=model.config.hidden_size
    activations = []
    logits = []
    generations = []
    module_hooks = [(layer, get_add_control_vector_all_hook(-a*steering_vector[:, i*model_dim:(i+1)*model_dim])) for i, layer in enumerate(model.transformer.h) if i in layers] # use negative scaling factor due to ordering of the examples when creating the steering vector
    
    for i, input_ids in enumerate(all_input_ids):
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        activations_example = []
        inst_module_hooks = module_hooks + [(layer, get_extract_last_activation_hook(activations_example)) for layer in model.transformer.h]

        logging.info(f"Extracting instruction logits for example {i}")
        with add_forward_hooks(inst_module_hooks):
            outputs = model(input_ids)
            logits.append(outputs.logits[0, -1, :])
        
        logging.info(f"Generating text for example {i}")
        with add_forward_hooks(module_hooks):
            generated_tokens = model.generate(input_ids, generation_config=generation_config)
        
        generations.append(generated_tokens[0])
        activations_cached = [act.to("cuda:0") for act in activations_example] # move all activations to the same GPU, sketchy solution, TODO: fix later
        activations_all_layers = torch.cat(activations_cached, dim=0) # (num_layers * model_dim)
        activations.append(activations_all_layers)

    logits = torch.stack([logits.to("cuda:0") for logits in logits], dim=0)
    activations = torch.stack(activations, dim=0)
    return generations, logits, activations