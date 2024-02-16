import torch
import torch.nn as nn
from nncore.nn import (MODELS, build_linear_modules, build_model, build_act_layer, kaiming_init_,
                       build_norm_layer, MultiHeadAttention, Sequential, ModuleList, FeedForwardNetwork)
import torch.nn.functional as F

class ReadFeedForwardNetwork(nn.Module):
    """
    Feed Forward Network introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        ratio (float, optional): The ratio of hidden layer dimensions with
            respect to the input dimensions. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 ratio=4,
                 p=0.1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 rank=32):
        super(ReadFeedForwardNetwork, self).__init__()

        self._dims = dims
        self._ratio = ratio
        self._p = p
        self._h_dims = int(dims * ratio)
        
        self.first_linear_mapping_downprojection = nn.Linear(dims, rank)
        self.first_linear_mapping_upprojection = nn.Linear(rank, self._h_dims)
        self.first_act_layer = build_act_layer(act_cfg)
        self.first_norm_layer = build_norm_layer('drop', p=p)
        self.second_linear_mapping_downprojection = nn.Linear(self._h_dims, rank)
        kernel_size = 3
        self.rnn = nn.RNN(rank, rank, batch_first=True)

        self.second_linear_mapping_upprojection = nn.Linear(rank, self._dims)
        self.second_norm_layer = build_norm_layer('drop', p=p)

        torch.nn.init.normal_(self.rnn.weight_ih_l0.data, 0.0, 1e-3)
        torch.nn.init.normal_(self.rnn.bias_ih_l0.data, 0.0, 1e-3)
        torch.nn.init.normal_(self.rnn.weight_hh_l0.data, 0.0, 1e-3)
        torch.nn.init.normal_(self.rnn.bias_hh_l0.data, 0.0, 1e-3)

        self.gelu = nn.GELU()

    def __repr__(self):
        return '{}(dims={}, ratio={}, p={})'.format(self.__class__.__name__,
                                                    self._dims, self._ratio,
                                                    self._p)

    def reset_parameters(self):
        kaiming_init_(self.first_linear_mapping_downprojection)
        kaiming_init_(self.first_linear_mapping_upprojection)
        kaiming_init_(self.second_linear_mapping_downprojection)
        kaiming_init_(self.second_linear_mapping_upprojection)
        # for m in self.mapping:
        #     if isinstance(m, nn.Linear):
        #         kaiming_init_(m)

    def forward(self, x):
        x_original = x
        output = self.first_linear_mapping_downprojection(x)
        output = self.first_act_layer(output)
        output = self.first_linear_mapping_upprojection(output)
        output = self.first_norm_layer(output)
        output = self.second_linear_mapping_downprojection(output)
        output = self.rnn(output)[0]
        output = self.gelu(output)
        output = self.second_linear_mapping_upprojection(output)
        output = self.second_norm_layer(output)

        # output = self.gelu(self.transformer_linear(output))

        output = x_original + output
        return output


class ReadDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer introduced in [1].

    Args:
        dims (int): The input feature dimensions.
        heads (int, optional): The number of attention heads. Default: ``1``.
        ratio (int, optional): The ratio of hidden layer dimensions in the
            feed forward network. Default: ``1``.
        p (float, optional): The dropout probability. Default: ``0.1``.
        pre_norm (bool, optional): Whether to apply the normalization before
            instead of after each layer. Default: ``True``.
        norm_cfg (dict | str | None, optional): The config or name of the
            normalization layer. Default: ``dict(type='LN')``.
        act_cfg (dict | str | None, optional): The config or name of the
            activation layer. Default: ``dict(type='ReLU', inplace=True)``.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 pre_norm=True,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ReadDecoderLayer, self).__init__()

        self._dims = dims
        self._heads = heads
        self._ratio = ratio
        self._p = p
        self._pre_norm = pre_norm

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.ffn = ReadFeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, x, mem, q_pe=None, k_pe=None, mask=None):
        if self._pre_norm:
            v = self.norm1(x)
            q = k = v if q_pe is None else v + q_pe
            d = self.att1(q, k, v, mask=mask)
            x = x + d

            q = self.norm2(x)
            q = q if q_pe is None else q + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = x + d

            d = self.ffn(d)
            d = self.norm3(x)
            x = x + d
        else:
            q = k = x if q_pe is None else x + q_pe
            d = self.att1(q, k, x, mask=mask)
            x = self.norm1(x + d)

            q = x if q_pe is None else x + q_pe
            k = mem if k_pe is None else mem + k_pe
            d = self.att2(q, k, mem, mask=mask)
            x = self.norm2(x + d)

            d = self.ffn(x)
            x = self.norm3(x + d)

        return x

@MODELS.register()
class QueryGenerator(nn.Module):

    def __init__(self, dims=None, p=0.3, enc_cfg=None, **kwargs):
        super(QueryGenerator, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = build_norm_layer(drop_cfg)
        self.mapping = build_linear_modules(dims, **kwargs)
        self.encoder = build_model(enc_cfg, enc_dims)

    def forward(self, x, mem=None, **kwargs):
        if mem is None:
            mem = x.new_zeros(x.size(0), 10, x.size(2))
        mask = torch.where(mem[:, :, 0].isfinite(), 1, 0)
        mem[~mem.isfinite()] = 0
        if self.dropout is not None:
            mem = self.dropout(mem)
        if self.mapping is not None:
            mem = self.mapping(mem)
        if self.encoder is not None:
            x = self.encoder(x, mem, mask=mask, **kwargs)
        return x


@MODELS.register()
class QueryDecoder(nn.Module):

    def __init__(self, dims=None, pos_cfg=None, dec_cfg=None, norm_cfg=None):
        super(QueryDecoder, self).__init__()

        self.q_pos_enc = build_model(pos_cfg, dims)
        self.k_pos_enc = build_model(pos_cfg, dims)
        self.decoder = ModuleList([ReadDecoderLayer(dims)])
        self.norm = build_norm_layer(norm_cfg, dims)

    def forward(self, x, mem=None, **kwargs):
        out = [x]
        if self.decoder is not None:
            q_pe = None if self.q_pos_enc is None else self.q_pos_enc(x)
            k_pe = None if self.k_pos_enc is None or mem is None else self.k_pos_enc(x)
            for dec in self.decoder:
                hid = dec(out[-1], mem=mem, q_pe=q_pe, k_pe=k_pe, **kwargs)
                out.append(hid)
        x = out if len(out) == 1 else out[1:]
        if self.norm is not None:
            x = [self.norm(h) for h in x]
        return x
