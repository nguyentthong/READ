import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class ReadParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, read_dropout_p=0.0, read_alpha=1):
        super().__init__()
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.read_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.read_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.read_A, a=math.sqrt(5))
        self.read_alpha, self.rank = read_alpha, rank
        self.scaling = read_alpha / rank
        self.read_dropout = nn.Dropout(p=read_dropout_p) if read_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if read_dropout_p > 0 else lambda x: x
        self.register_buffer("read_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.read_A.dtype))
        self.forward_fn = self.read_forward

    def _dropout(self, A):
        return A * self.read_dropout(self.read_dropout_mask)

    def read_forward(self, X):
        return X + torch.matmul(*self.swap((self.read_B, self.dropout_fn(self.read_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_read(self):
        self.forward_fn = lambda x: x

    def enable_read(self):
        self.forward_fn = self.read_forward

    @classmethod
    def from_linear(cls, layer, rank=4, read_dropout_p=0.0, read_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, read_dropout_p=read_dropout_p, read_alpha=read_alpha
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, read_dropout_p=0.0, read_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, read_dropout_p=read_dropout_p, read_alpha=read_alpha
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, read_dropout_p=0.0, read_alpha=1):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, read_dropout_p=read_dropout_p, read_alpha=read_alpha
        )


default_read_config = {  
    nn.Linear: {
        "weight": partial(ReadParametrization.from_linear, rank=4),
    },
}


def apply_read(layer, register=True, merge=False, read_config=default_read_config):
    """add read parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in read_config:
            for attr_name, parametrization in read_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else: 
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_read(model, read_config=default_read_config):
    """add read parametrization to all layers in a model. Calling it twice will add read twice"""
    model.apply(partial(apply_read, read_config=read_config))


def add_read_by_name(model, target_module_names, read_config=default_read_config):
    """Add read parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_read(layer, read_config=read_config)


def merge_read(model):
    """merge read parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_read, register=False, merge=True))


def remove_read(model):
    """remove read parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_read, register=False, merge=False))
