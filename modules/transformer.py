

import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy

from .relative_transformer import RelativeMultiHeadAttn

import numpy as np


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """

        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        """

        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size, dropout=0.3, scaled=False, temper=1):
        super(KeyValueMemoryNetwork, self).__init__()
        self.key_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx=0)
        self.scaled = scaled
        self.scale = np.power(emb_size, 0.5 * temper)
        self.softmax = nn.Softmax(dim=2)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.Tensor(emb_size, emb_size))

    def forward(self, key_seq, value_seq, hidden, mask_matrix, nan_matrix):
        """
        :param key_seq: word_seq: batch * seq_len
        :param value_seq: word_pos_seq: batch * seq_len
        :param hidden: batch * seq_len * hidden
        :param mask_matrix: batch * seq_len * seq_len
        :return:
        """
        key_embed = self.key_embedding(key_seq)
        value_embed = self.value_embedding(value_seq)

        # Q*K^T
        u = torch.bmm(hidden.matmul(self.weight), key_embed.transpose(1, 2))
        # u = torch.bmm(hidden, key_embed.transpose(1, 2))
        u = u / self.scale

        # softmax
        mask_matrix = torch.clamp(mask_matrix.float(), 0, 1)
        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, mask_matrix)
        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        # attn_score * V
        o = torch.bmm(p, value_embed)
        return o


class GatingMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GatingMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = input.mul(gate) + hidden.mul(1 - gate)
        return output
#
#
# class FusionAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(FusionAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.alignment_network = nn.Linear(hidden_size, 1)
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, input, hidden):
#         batch_size, seq_len, _ = input.shape
#         # b * l * 2 * n
#         combined = torch.cat([input.unsqueeze(2), hidden.unsqueeze(2)], dim=2)
#         # b * l * 2 * 1
#         attn_score = self.alignment_network(combined)
#         # b * l * 2
#         attn_score = attn_score.squeeze(3)
#         # b * l * 2
#         normalized_score = self.softmax(attn_score)
#         # bl * 2
#         normalized_score = normalized_score.view(-1, 2)
#         # bl * 1 * 2
#         normalized_score = normalized_score.unsqueeze(1)
#         # bl * 2 * n
#         value = combined.view(-1, 2, self.hidden_size)
#         # bl * 1 * n
#         output = torch.bmm(normalized_score, value)
#         # b * l * n
#         output = output.view(batch_size, seq_len, -1)
#         return output


class AdaptedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None,
                 kv_vocab_size=None,
                 kv_feature_vocab_size=None,
                 kv_emb_size=None,
                 kv_attn_dropout=None,
                 kv_dropout=None,
                 kv_scaled=True,
                 gate_dropout=None
                 ):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        self.kv_memory = KeyValueMemoryNetwork(
            vocab_size=kv_vocab_size,
            feature_vocab_size=kv_feature_vocab_size,
            emb_size=kv_emb_size,
            dropout=kv_attn_dropout,
            scaled=kv_scaled
        )
        self.gating = GatingMechanism(hidden_size=kv_emb_size)

        self.memory_dropout = nn.Dropout(p=kv_dropout)
        self.fusion_attn = FusionAttention(hidden_size=kv_emb_size)
        self.gate_dropout = nn.Dropout(p=gate_dropout)

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, mask, additional_tuple):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        chars, features, pos_matrix, nan_matrix = additional_tuple

        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask)
            y = self.memory_dropout(self.kv_memory(chars, features, x, pos_matrix, nan_matrix))
            x = self.fusion_attn(x, y)
            x = self.gate_dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask)
        return x


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)