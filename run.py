import os

nums = [10]
device = "0"
dataset = "WNUT_17_train"
seed = 14
attn_type = "dot"
fusion_type = "gate-concat"
bert_model = "../data/bert-large-cased"
glove_path = "../data/glove.100d.txt"
pool_method = "first"

for i in nums:
    log = "log/{}_{}_bert_elmo_{}.txt".format(dataset, pool_method, i)
    os.system("CUDA_VISIBLE_DEVICES={} python3 train_bert_elmo_en.py --dataset {} "
              "--seed {} --kv_attn_type {} --fusion_type {} --context_num {} "
              "--bert_model {} --pool_method {} --glove_path {} "
              "--lr 0.0001 --trans_dropout 0.2 --fc_dropout 0.4 --memory_dropout 0.2 "
              "--fusion_dropout 0.2 --log {}".format(device, dataset, seed, attn_type, fusion_type,
                                                     i, bert_model, pool_method, glove_path, log))
