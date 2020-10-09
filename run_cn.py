import os

num = 10
dataset = "weibo"
seed = 14
attn_type = "dot"
fusion_type = "gate-concat"
bert_model = "data/bert-base-chinese"
glove_path = "data/tencent_unigram.txt"
pool_method = "first"
zen_model = "zen_base/"

log = "log/{}_zen_{}_embed_attn_context_{}.txt".format(dataset, pool_method, num)
os.system("python3 train_zen_cn.py --dataset {} "
          "--seed {} --kv_attn_type {} --fusion_type {} --context_num {} "
          "--bert_model {} --pool_method {} --glove_path {} "
          "--zen_model {} "
          "--lr 0.0001 --trans_dropout 0.2 --fc_dropout 0.4 --memory_dropout 0.2 "
          "--fusion_dropout 0.2 --log {}".format(dataset, seed, attn_type, fusion_type,
                                                 num, bert_model, pool_method, glove_path, zen_model, log))