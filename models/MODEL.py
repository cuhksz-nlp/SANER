
import numpy as np
from backend.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, AdaptedTransformerEncoder

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Callable


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class KeyValueMemoryNetwork(nn.Module):
    def __init__(self, vocab_size, feature_vocab_size, emb_size, scaled=False, temper=1, attn_type="dot", use_key=True):
        super(KeyValueMemoryNetwork, self).__init__()
        self.use_key = use_key
        if self.use_key:
            self.key_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.value_embedding = nn.Embedding(feature_vocab_size, emb_size, padding_idx=0)
        self.scaled = scaled
        self.scale = np.power(emb_size, 0.5 * temper)
        self.attn_type = attn_type
        if attn_type == "bilinear":
            self.weight = nn.Parameter(torch.Tensor(emb_size, emb_size))

    def forward(self, value_seq, hidden):
        """
        :param key_seq: word_seq: batch * seq_len
        :param value_seq: word_pos_seq: batch * seq_len
        :param hidden: batch * seq_len * hidden
        :param mask_matrix: batch * seq_len * seq_len
        :return:
        """
        value_embed = self.value_embedding(value_seq)

        batch_size, seq_len, context_num, dim = value_embed.shape
        value_embed = value_embed.view(batch_size * seq_len, context_num, dim)
        hidden = hidden.view(batch_size * seq_len, 1, dim)

        # attn_score = Q*(K^T)
        if self.attn_type == "dot":
            u = torch.bmm(hidden, value_embed.transpose(1, 2))
        elif self.attn_type == "bilinear":
            u = torch.bmm(hidden.matmul(self.weight), value_embed.transpose(1, 2))
        if self.scaled:
            u = u / self.scale

        # softmax
        # mask_matrix = torch.clamp(mask_matrix.float(), 0, 1)
        # exp_u = torch.exp(u)
        # delta_exp_u = torch.mul(exp_u, mask_matrix)
        # sum_delta_exp_u1 = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        # p = torch.div(delta_exp_u, sum_delta_exp_u1 + 1e-10)

        p = torch.softmax(u, dim=-1)

        o = torch.bmm(p, value_embed)
        o = o.view(batch_size, seq_len, dim)
        return o


class GateConcMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(GateConcMechanism, self).__init__()
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
        # input: hidden state from encoder; hidden: hidden state from key value memory network
        # output = [gate * input; (1 - gate) * hidden]
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = torch.cat([input.mul(gate), hidden.mul(1 - gate)], dim=2)
        return output


class LinearGateAddMechanism(nn.Module):
    def __init__(self, hidden_size=None):
        super(LinearGateAddMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
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
        # input: hidden state from encoder; hidden: hidden state from key value memory network
        # output = [gate * input; (1 - gate) * hidden]
        input = self.linear1(input)
        hidden = self.linear2(hidden)
        gated = input.matmul(self.w1.t()) + hidden.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        output = input.mul(gate) + hidden.mul(1 - gate)
        return output


style_map = {
    'add': lambda x, y: x + y,
    'concat': lambda *args: torch.cat(args, args[0].dim() - 1),
    'diff': lambda x, y: x - y,
    'abs-diff': lambda x, y: torch.abs(x - y),
    'concat-diff': lambda x, y: torch.cat((x, y, x - y), x.dim() - 1),
    'concat-add': lambda x, y: torch.cat((x, y, x + y), x.dim() - 1),
    'concat-abs-diff': lambda x, y: torch.cat((x, y, torch.abs(x - y)), x.dim() - 1),
    'mul': lambda x, y: torch.mul(x, y),
    'concat-mul-diff': lambda x, y: torch.cat((x, y, torch.mul(x, y), torch.abs(x - y)), x.dim() - 1)
}


class FusionModule(nn.Module):
    """
    FusionModule定义了encoder output与kv output之间的信息融合方式
    """
    def __init__(self, layer=1, fusion_type="concat", input_size=1024, output_size=1024, dropout=0.2):
        """
        :param layer: layer代表highway的层数
        :param fusion_type: fusion_type代表融合方式
        :param size: size代表输出dimension
        :param dropout: 代表fusion之后，highway之前的dropout
        """
        super(FusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.layer = layer
        if self.layer > 0:
            self.highway = Highway(size=output_size, num_layers=layer, f=torch.nn.functional.relu)
        # if self.fusion_type == "gate-add":
        #     self.gate = GateAddMechanism(hidden_size=input_size)
        elif self.fusion_type == "gate-concat":
            self.gate = GateConcMechanism(hidden_size=input_size)
        elif self.fusion_type == "l-gate-add":
            self.gate = LinearGateAddMechanism(hidden_size=input_size)
        self.fusion_dropout = nn.Dropout(p=dropout)

    def forward(self, enc_out, kv_out):
        # 如果使用gate的方式进行fusion
        if self.fusion_type in ["gate-add", "gate-concat", "l-gate-add"]:
            fused = self.gate(enc_out, kv_out)
        # 直接用concat或者add等方式进行fusion
        else:
            fused = style_map[self.fusion_type](enc_out, kv_out)
        fused = self.fusion_dropout(fused)
        # 进行highway操作
        if self.layer > 0:
            fused = self.highway(fused)
        return fused


_dim_map = {
    "concat": 2,
    "gate-concat": 2,
}


class ModelClass(nn.Module):
    def __init__(self, tag_vocab, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None,
                 use_knowledge=True,
                 vocab_size=None,
                 feature_vocab_size=None,
                 kv_attn_type="dot",
                 memory_dropout=0.2,
                 fusion_dropout=0.2,
                 fusion_type='concat',
                 highway_layer=0
                 ):
        """
        :param tag_vocab: backend Vocabulary
        :param embed: backend TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        :param use_knowledge: 是否使用stanford corenlp的知识
        :param feature2count: 字典, {"gram2count": dict, "pos_tag2count": dict, "chunk_tag2count": dict, "dep_tag2count": dict},
        :param
        """
        super().__init__()
        self.use_knowledge = use_knowledge
        self.vocab_size = vocab_size
        self.feature_vocab_size = feature_vocab_size

        self.embed = embed
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)

        self.kv_memory = KeyValueMemoryNetwork(vocab_size=vocab_size, feature_vocab_size=feature_vocab_size,
                                               attn_type=kv_attn_type, emb_size=d_model, scaled=True)

        self.output_dim = d_model * _dim_map[fusion_type]
        self.fusion = FusionModule(fusion_type=fusion_type, layer=highway_layer, input_size=d_model,
                                   output_size=self.output_dim, dropout=fusion_dropout)

        self.memory_dropout = nn.Dropout(p=memory_dropout)
        self.out_fc = nn.Linear(self.output_dim, len(tag_vocab))
        self.fc_dropout = nn.Dropout(fc_dropout)

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, target, bigrams=None, features=None):
        # get the hidden state from transformer encoder
        mask = chars.ne(0)
        hidden = self.embed(chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            hidden = torch.cat([hidden, bigrams], dim=-1)
        hidden = self.in_fc(hidden)
        # additional_tuple is used for layer-wise fusion
        # additional_tuple = (chars, features, pos_matrix, nan_matrix)
        encoder_output = self.transformer(hidden, mask)
        # new add
        # kv_output: hidden state of key value memory network
        kv_output = self.kv_memory(features, hidden)
        kv_output = self.memory_dropout(kv_output)
        # o: output of gating mechanism
        concat = self.fusion(encoder_output, kv_output)

        concat = self.fc_dropout(concat)
        concat = self.out_fc(concat)
        logits = F.log_softmax(concat, dim=-1)
        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}

    def forward(self, chars, target, bigrams=None, features=None):
        return self._forward(chars, target, bigrams, features)

    def predict(self, chars, bigrams=None, features=None):
        return self._forward(chars, target=None, bigrams=bigrams, features=features)
