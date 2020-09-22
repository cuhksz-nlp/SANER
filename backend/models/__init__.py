"""
backend 在 :mod:`~backend.models` 模块中内置了如 :class:`~backend.models.CNNText` 、
:class:`~backend.models.SeqLabeling` 等完整的模型，以供用户直接使用。

.. todo::
    这些模型的介绍（与主页一致）


"""
__all__ = [
    "CNNText",
    
    "SeqLabeling",
    "AdvSeqLabel",
    "BiLSTMCRF",
    
    "ESIM",
    
    "StarTransEnc",
    "STSeqLabel",
    "STNLICls",
    "STSeqCls",
    
    "BiaffineParser",
    "GraphParser",

    "BertForSequenceClassification",
    "BertForSentenceMatching",
    "BertForMultipleChoice",
    "BertForTokenClassification",
    "BertForQuestionAnswering"
]

from .base_model import BaseModel
from .bert import BertForMultipleChoice, BertForQuestionAnswering, BertForSequenceClassification, \
    BertForTokenClassification, BertForSentenceMatching
from .biaffine_parser import BiaffineParser, GraphParser
from .cnn_text_classification import CNNText
from .sequence_labeling import SeqLabeling, AdvSeqLabel, BiLSTMCRF
from .snli import ESIM
from .star_transformer import StarTransEnc, STSeqCls, STNLICls, STSeqLabel

import sys
from ..doc_utils import doc_process
doc_process(sys.modules[__name__])