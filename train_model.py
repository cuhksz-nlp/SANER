from models.MODEL import ModelClass
from backend.embeddings import CNNCharEmbedding
from backend import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from backend import SpanFPreRecMetric, BucketSampler
from backend.io.pipe.conll import OntoNotesNERPipe
from backend.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding, ElmoEmbedding, BertEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
from modules.pipe import ENNERPipe
from modules.callbacks import EvaluateCallback

from augmentation import get_neighbor_for_vocab, build_instances

import os
import argparse
from datetime import datetime

import random
import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(14)

dataset = "sampleSet"

n_heads = 12
head_dims = 128
num_layers = 2
lr = 0.0001
attn_type = 'adatrans'
optim_type = 'adam'
trans_dropout = 0.2
batch_size = 32


char_type = 'adatrans'

pos_embed = None

model_type = 'pure'
warmup_steps = 0.01
after_norm = 1
fc_dropout = 0.2
normalize_embed = True

encoding_type = 'bioes'
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)
context_num = 10

# glove_download_path = "/Users/nyy/.fastNLP/embedding/glove.6B.100d/glove.6B.100d.txt"
# glove_path = "/Users/nyy/.fastNLP/embedding/glove.6B.100d/glove.context.txt"

device = "cpu"

feature_level = "all"
memory_dropout = 0.2
fusion_dropout = 0.2
kv_attn_type = "dot"
fusion_type = "gate-concat"
highway_layer = 0


def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])


save_path = "ckpt/elmo_{}_{}_{}_{}.pth".format(dataset, model_type, context_num, print_time())
save_path = None

logPath = "log/log_{}_{}_{}.txt".format(dataset, context_num, print_time())


def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")

def load_data():

    paths = {"train": "data/sampleSet/train.txt", "test": "data/sampleSet/test.txt", "dev": "data/sampleSet/dev.txt"}
    data = ENNERPipe(encoding_type=encoding_type).process_from_file(paths)

    dict_save_path = os.path.join("data/sampleSet/data.pth")
    context_dict, context_word2id, context_id2word = get_neighbor_for_vocab(
        data.get_vocab('words').word2idx, "/Users/nyy/.backend/embedding/glove.6B.100d/glove.context.txt", "/Users/nyy/.backend/embedding/glove.6B.100d/glove.6B.100d.txt", dict_save_path
    )
    train_feature_data, test_feature_data, dev_feature_data = build_instances(
        "data/{}".format(dataset), context_num, context_dict
    )

    word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                 model_dir_or_name='en-glove-6b-100d',
                                 requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                 only_norm_found_vector=normalize_embed)

    embed = StackEmbedding([word_embed], dropout=0, word_dropout=0.02)

    data.rename_field('words', 'chars')

    return data, embed, train_feature_data, test_feature_data, dev_feature_data, context_word2id, context_id2word


data_bundle, embed, train_feature_data, test_feature_data, dev_feature_data, feature2id, id2feature = load_data()

vocab_size = len(data_bundle.get_vocab('chars'))
feature_vocab_size = len(feature2id)

model = ModelClass(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
              d_model=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=trans_dropout,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=None,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=attn_type=='naive',
              vocab_size=vocab_size,
              feature_vocab_size=feature_vocab_size,
              kv_attn_type=kv_attn_type,
              memory_dropout=memory_dropout,
              fusion_dropout=fusion_dropout,
              fusion_type=fusion_type,
              highway_layer=highway_layer
              )

if optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=data_bundle.get_dataset('test'),
                                     test_feature_data=test_feature_data,
                                     feature2id=feature2id,
                                     id2feature=id2feature,
                                     context_num=context_num
                                     )

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=0, n_epochs=100, dev_data=data_bundle.get_dataset('test'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  use_knowledge=True,
                  train_feature_data=train_feature_data,
                  test_feature_data=test_feature_data,
                  feature2id=feature2id,
                  id2feature=id2feature,
                  logger_func=write_log,
                  context_num=context_num
                  )

trainer.train(load_best_model=False)
