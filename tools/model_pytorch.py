import os
import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class OpTarget:
    key: str
    dim: int


def split_model(model_dict, targets: Tuple[OpTarget], tp_size: int) -> Tuple[dict]:
    '''Split the model dict into N(equals tp_size) partitions according to targets.
    Arguments:
        model_dic: input model dict with key:tensor format.
        targets: tuple of split key and dim.
                it contains all the split tensor and which dim.
                
                the struct is:
                [
                {key: "layers.30.attention.wq.weight", dim: -2},
                {key: "layers.30.attention.wk.weight", dim: -2},
                ......
                {key: "layers.30.feed_forward.w2.weight", dim: -1}
                ]
                
                - dim -1 means row split
                   e.g. XW W will split into:
                   -    -
                   | w1 |
                   | w2 |
                   |....|
                   | wn |
                   -    -
                - dim -2 means column split
                   e.g. XW W will split into:
                   -                 -
                   | w1, w2, ..., wn |
                   -                 -
        tp_size: number of partitions.
    '''

    assert isinstance(model_dict, dict), f"the input {type(model_dict)} is not the dict type."
    assert all(t.key in model_dict for t in targets), f"target key cannot find in weight"    
    
    tensors = {t.key: split_tensor(model_dict[t.key], t.dim, tp_size) for t in targets}
    results = [{key : tensors[key][i] if key in tensors else model_dict[key].clone() for key in model_dict.keys()} for i in range(tp_size)]

    return results 

def clone_tensor(weight: torch.Tensor, time: int) -> Tuple[torch.Tensor,...]:
    '''Clone weith time times.
    Arguments:
        weight: target weight
        time: times
    '''

    return [weight.clone() for i in range(time)]

def split_tensor(weight: torch.Tensor, dim: int, tp_size: int) -> Tuple[torch.Tensor,...]:
    '''Split the weight into N smaller weights along the dim.
    Arguments:
        weight: input Tensor
        dim: split dimension.
        tp_size: output number of weights.
    '''

    assert -weight.dim() <= dim and dim < weight.dim(), f"invalid input dim({dim}), out of the weight's dims({weight.dim()})"
    assert weight.shape[dim] % tp_size == 0, f"input weights's size({weight.shape[dim]}) on dim({dim}) can't be divisible by tp_size({tp_size})."

    tensor_list = torch.split(weight, weight.shape[dim] // tp_size, dim=dim)
    #Note: the torch.split will return tensor without contiguous.
    #      but contiguous will keep the original data.
    #      call clone to make it clean.
    out = [t.clone() for t in tensor_list]

    return out

