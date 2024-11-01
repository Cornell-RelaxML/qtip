"""Simple sharded model
"""
import glog
import torch
from torch import nn

from lib.linear.quantized_linear import QuantizedLinear

@torch.compile
def convert_args(args, kwargs, device, dtype):

    def convert_tensor(tensor):
        return tensor.to(device, non_blocking=True)

    dev_args = []
    for i in range(len(args)):
        dev_args.append(
            convert_tensor(args[i]) if isinstance(args[i], torch.Tensor) else args[i])
        
    for i in kwargs:
        if isinstance(kwargs[i], torch.Tensor):
            kwargs[i] = convert_tensor(kwargs[i])
    if 'position_embeddings' in kwargs:
        kwargs['position_embeddings'] = (
            kwargs['position_embeddings'][0].to(device, non_blocking=True),
            kwargs['position_embeddings'][1].to(device, non_blocking=True))
    return dev_args, kwargs


class Shard(nn.Module):

    def __init__(self, layers, arg_fn):
        super().__init__()
        self.layers = layers
        self.arg_fn = arg_fn

    def forward(self, *args, **kwargs):
        for layer in self.layers:
            output = layer(*args, **kwargs)
            args, kwargs = self.arg_fn(output, args, kwargs)
        return args, kwargs


class ShardDecoderLayers(nn.Module):

    def __init__(self,
                 shards,
                 dtype):
        super().__init__()

        # shards is list of [(device, arg_fn, modulelist)]

        self.shards = nn.ModuleList([_['shard'] for _ in shards])
        self.devices = [_['device'] for _ in shards]

        for i in range(len(shards)):
            device = self.devices[i]
            self.shards[i] = Shard(self.shards[i],
                                   shards[i]['arg_fn']).to(device)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        for i in range(len(self.shards)):
            device = self.devices[i]
            args, kwargs = convert_args(args, kwargs, device, self.dtype)
            args, kwargs = self.shards[i](*args, **kwargs)

        return args[0], kwargs['past_key_value']
