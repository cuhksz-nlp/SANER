from models.TENER import TENER
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.embeddings import StackEmbedding, ElmoEmbedding, BertEmbedding
from modules.pipe import WNUT_17NERPipe
from modules.callbacks import EvaluateCallback

from get_context import get_neighbor_for_vocab, build_instances

import os
import argparse
from datetime import datetime

import random
import numpy as np
import torch


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='sampleSet')
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--context_num', type=int, default=10)
parser.add_argument('--glove_path', type=str)
parser.add_argument('--bert_model', type=str, required=True)
parser.add_argument('--pool_method', type=str, default="first", choices=["first", "last", "avg", "max"])
parser.add_argument('--trans_dropout', type=float, default=0.2)
parser.add_argument('--fc_dropout', type=float, default=0.4)
parser.add_argument('--memory_dropout', type=float, default=0.2)
parser.add_argument('--fusion_type', type=str, default='gate-concat',
                    choices=['concat', 'add', 'concat-add', 'gate-add', 'gate-concat'])
parser.add_argument('--fusion_dropout', type=float, default=0.2)
parser.add_argument('--highway_layer', type=int, default=0)
parser.add_argument('--kv_attn_type', type=str, default='dot')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)

dataset = args.dataset

n_heads = 12
head_dims = 128
num_layers = 2
lr = args.lr
attn_type = 'adatrans'
optim_type = 'adam'
trans_dropout = args.trans_dropout
batch_size = 32

char_type = 'adatrans'
embed_size = 30

# positional_embedding
pos_embed = None

model_type = 'elmo'
elmo_model = "en-original"
warmup_steps = 0.01
after_norm = 1
fc_dropout = args.fc_dropout
normalize_embed = True

encoding_type = 'bioes'
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)
context_num = args.context_num
glove_path = args.glove_path

# new_add
feature_level = "all"
memory_dropout = args.memory_dropout
fusion_dropout = args.fusion_dropout
kv_attn_type = args.kv_attn_type
fusion_type = args.fusion_type
highway_layer = args.highway_layer


def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])


name = 'caches/{}_{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type,
                                                     normalize_embed, context_num)
save_path = "ckpt/{}_{}_{}_{}.pth".format(dataset, model_type, context_num, print_time())

logPath = args.log
if not args.log:
    logPath = "log/log_{}_{}_{}.txt".format(dataset, context_num, print_time())


def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")


@cache_results(name, _refresh=False)
def load_data():

    paths = {
        "train": "../data/{}/train.txt".format(dataset),
        "test": "../data/{}/test.txt".format(dataset),
        "dev": "../data/{}/dev.txt".format(dataset)
    }
    data = WNUT_17NERPipe(encoding_type=encoding_type).process_from_file(paths)

    dict_save_path = os.path.join("../data/{}/data.pth".format(dataset))
    context_dict, context_word2id, context_id2word = get_neighbor_for_vocab(
        data.get_vocab('words').word2idx, glove_path, dict_save_path
    )

    train_feature_data, dev_feature_data, test_feature_data = build_instances(
        "../data/{}".format(dataset), context_num, context_dict
    )

    data.rename_field('words', 'chars')
    embed = ElmoEmbedding(vocab=data.get_vocab('chars'), model_dir_or_name=elmo_model, layers='mix', requires_grad=False,
                          word_dropout=0.0, dropout=0.5, cache_word_reprs=False)
    embed.set_mix_weights_requires_grad()
    bert_embed = BertEmbedding(vocab=data.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=False)
    embed = StackEmbedding([embed, bert_embed], dropout=0, word_dropout=0.02)

    return data, embed, train_feature_data, dev_feature_data, test_feature_data, context_word2id, context_id2word


data_bundle, embed, train_feature_data, dev_feature_data, test_feature_data, feature2id, id2feature = load_data()

train_data = list(data_bundle.get_dataset("train"))
dev_data = list(data_bundle.get_dataset("dev"))
test_data = list(data_bundle.get_dataset("test"))

vocab_size = len(data_bundle.get_vocab('chars'))
feature_vocab_size = len(feature2id)

model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
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
                  num_workers=0, n_epochs=50, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  use_knowledge=True,
                  train_feature_data=train_feature_data,
                  test_feature_data=dev_feature_data,
                  feature2id=feature2id,
                  id2feature=id2feature,
                  logger_func=write_log,
                  context_num=context_num
                  )

trainer.train(load_best_model=False)
