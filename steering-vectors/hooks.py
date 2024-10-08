from contextlib import contextmanager
from typing import List, Tuple, Callable
import torch


@contextmanager
def add_forward_hooks(forward_hooks: List[Tuple[torch.nn.Module, Callable]]):
  """Context manager for adding forward hooks to a list of modules.

  Approach to add PyTorch hooks to each layer inspired by the approach taken in 
  https://github.com/andyrdt/refusal_direction ([3] in the thesis).

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

def get_extract_mean_activation_hook(activations: List[torch.Tensor]):
  """The hook extracts the mean activations over the whole output sequence produced by the
  component and stores them in the activations cache provided as input. The output parameter
  of the hook is the activations tuple produced by the component where the hook is added. The
  first item of the tuple is the activations tensor which is of shape (batch, sequence, hidden
  dimension).

  Args:
      activations (List[torch.Tensor]): List of tensors where the activations are stored.
  """
  def hook(module, input, output):
    assert len(output) == 2 # For Bloom architecture
    assert len(output[0].shape) == 3 # For Bloom architecture
    activations.append(output[0][0, :, :].mean(dim=0))
  return hook

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