from minread import ReadParametrization
from torch import nn


def apply_to_read(fn):
    def apply_fn(layer):
        if isinstance(layer, ReadParametrization):
            fn(layer)

    return apply_fn


enable_read = lambda model: model.apply(apply_to_read(lambda x: x.enable_read()))
disable_read = lambda model: model.apply(apply_to_read(lambda x: x.disable_read()))



def name_is_read(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["read_A", "read_B"]
    )


def name_is_bias(name):
    return name.split(".")[-1] == "bias"


def get_params_by_name(model, print_shapes=False, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_read_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_read)


def get_bias_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias)


def get_read_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if name_is_read(k)}



def _prepare_for_multiple_read(read_layer):
    read_layer.read_As = []
    read_layer.read_Bs = []


def _append_read(read_layer):
    read_layer.read_As.append(nn.Parameter(read_layer.read_A.clone()))
    read_layer.read_Bs.append(nn.Parameter(read_layer.read_B.clone()))


def load_multiple_read(model, read_state_dicts):
    model.apply(apply_to_read(_prepare_for_multiple_read))
    for state_dict in read_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_read(_append_read))
    return model


def _select_read(read_layer, index):
    read_layer.read_A = read_layer.read_As[index]
    read_layer.read_B = read_layer.read_Bs[index]


def select_read(model, index):
    model.apply(apply_to_read(lambda x: _select_read(x, index)))
    return model



def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
    embedding.parametrizations.weight.original = linear.parametrizations.weight.original
    embedding.parametrizations.weight[0].read_A = linear.parametrizations.weight[0].read_B
    embedding.parametrizations.weight[0].read_B = linear.parametrizations.weight[0].read_A


def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
    embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
    embedding.parametrizations.weight[0].read_A = nn.Parameter(embedding.parametrizations.weight[0].read_A.clone())
    embedding.parametrizations.weight[0].read_B = nn.Parameter(embedding.parametrizations.weight[0].read_B.clone())
