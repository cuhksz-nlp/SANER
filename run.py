import os

num = 10
dataset = "sampleSet"
seed = 14
attn_type = "dot"
fusion_type = "gate-concat"
bert_model = "../data/bert-large-cased"
glove_path = "../data/glove.100d.txt"
pool_method = "first"

log = "log/{}_{}_bert_elmo_{}.txt".format(dataset, pool_method, num)
os.system("python3 train_bert_elmo_en.py --dataset {} "
          "--seed {} --kv_attn_type {} --fusion_type {} --context_num {} "
          "--bert_model {} --pool_method {} --glove_path {} "
          "--lr 0.0001 --trans_dropout 0.2 --fc_dropout 0.4 --memory_dropout 0.2 "
          "--fusion_dropout 0.2 --log {}".format(dataset, seed, attn_type, fusion_type,
                                                 num, bert_model, pool_method, glove_path, log))
