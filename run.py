import os

# the number of similar words
num = 10
# dataset name
dataset = "sampleSet"
# seed
seed = 14
attn_type = "dot"
fusion_type = "gate-concat"
# path of the bert model
bert_model = "data/bert-large-cased"
# path of the pre-trained embeddings for getting the similar words
glove_path = "data/glove.100d.txt"
pool_method = "first"

log = "log/{}_{}_{}.txt".format(dataset, pool_method, num)
os.system("python3 train_bert_elmo_en.py --dataset {} "
          "--seed {} --kv_attn_type {} --fusion_type {} --context_num {} "
          "--bert_model {} --pool_method {} --glove_path {} "
          "--lr 0.0001 --trans_dropout 0.2 --fc_dropout 0.4 --memory_dropout 0.2 "
          "--fusion_dropout 0.2 --log {}".format(dataset, seed, attn_type, fusion_type,
                                                 num, bert_model, pool_method, glove_path, log))
