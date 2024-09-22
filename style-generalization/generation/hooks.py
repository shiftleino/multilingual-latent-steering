from contextlib import contextmanager
from typing import List, Tuple, Callable
import torch


@contextmanager
def add_forward_hooks(forward_hooks: List[Tuple[torch.nn.Module, Callable]]):
  """Context manager for adding forward hooks to a list of modules.

  Args:
      forward_hooks (List[Tuple[torch.nn.Module, Callable]]): List of modules where the corresponding hooks are added.
  """
  try:
    handles = []
    for module, hook in forward_hooks:
      handle = module.register_forward_hook(hook)
      handles.append(handle)
    yield
  finally:
    for handle in handles:
      handle.remove()

def get_extract_last_activation_hook(activations: List[torch.Tensor]):
  """The hook extracts the activations of the last token in the sequence and stores them in the
  activations cache provided as input. The output parameter of the hook is the activations
  tuple produced by the component where the hook is added. The first item of the tuple is the 
  activations tensor which is of shape (batch, sequence, hidden dimension).
  
  Args:
      activations (List[torch.Tensor]): List of tensors where the activations are stored.
  """
  def hook(module, input, output):
    assert len(output) == 2
    assert len(output[0].shape) == 3
    activations.append(output[0][0, -1, :])
  return hook

def get_add_steering_vector_all_hook(steering_vector: torch.Tensor):
  """The hook adds a steering vector to the activations of all tokens in the sequence.
  The output parameter of the hook is the activations tuple produced by the component 
  where the hook is added. The first item of the tuple is the activations tensor which 
  is of shape (batch, sequence, hidden dimension)

  Args:
      Steering vector (torch.Tensor): Steering vector to be added to the activations.
  """
  def hook(module, input, output):
    steering_vector_gpu = steering_vector.to(output[0].device)
    output_activations = output[0] # -> (batch, seq_length, model_dim)
    modified_output = output_activations[0, :, :] + steering_vector_gpu
    batch_output_activations = output_activations[0, :, :] # -> (seq_length, model_dim)
    modified_output *= (torch.norm(batch_output_activations, p=2, dim=1).unsqueeze(1) / torch.norm(modified_output, p=2, dim=1).unsqueeze(1))
    output_activations[0, :, :] = modified_output
    return (output_activations, *output[1:])
  return hook
