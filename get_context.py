import os
import word2vec
from tqdm import tqdm
import torch

def get_neighbor_for_vocab(word2id, dataPath, dict_save_path):
    if os.path.exists(dict_save_path):
        data = torch.load(dict_save_path)
        return data.get("context_dict"), data.get("context_word2id"), data.get("context_id2word")
    model = word2vec.load(dataPath)
    ret_dict = {}
    vocab = set()

    for word in tqdm(word2id, desc="read vocab"):
        word_lower = word.lower()
        try:
            indices = model.similar(word_lower, n=20)[0]
        except Exception:
            continue
        ret = set([model.vocab[i] for i in indices])
        vocab |= ret
        ret_dict[word_lower] = ret

    index = 1
    context_word2id = {"[PAD]": 0}
    context_id2word = {0: "[PAD]"}

    for word in vocab:
        context_word2id[word] = index
        context_id2word[index] = word
        index += 1

    for word, context in ret_dict.items():
        context = [context_word2id[i] for i in context]
        ret_dict[word] = context

    save_dict = {
        "context_dict": ret_dict,
        "context_word2id": context_word2id,
        "context_id2word": context_id2word
    }
    torch.save(save_dict, dict_save_path)
    return ret_dict, context_word2id, context_id2word


def read_txt(file_path):
    sentence_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentence = []
        labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    sentence_list.append(sentence)
                    label_list.append(labels)
                    sentence = []
                    labels = []
                continue
            items = line.split()
            character = items[0]
            label = items[-1]
            sentence.append(character)
            labels.append(label)

    return sentence_list, label_list


def build_context(data_list, num, context_dict):
    ret_list = []
    for sent in tqdm(data_list, desc="build dataset"):
        ret = []
        for word in sent:
            word = word.lower()
            ret.append(context_dict.get(word, [0] * num)[:num])
        ret_list.append(ret)
    return ret_list


def build_instances(data_dir_path, num, context_dict):
    trainPath = os.path.join(data_dir_path, "train.txt")
    devPath = os.path.join(data_dir_path, "dev.txt")
    testPath = os.path.join(data_dir_path, "test.txt")

    train_list, _ = read_txt(trainPath)
    dev_list, _ = read_txt(devPath)
    test_list, _ = read_txt(testPath)

    train_feature_list = build_context(train_list, num, context_dict)
    dev_feature_list = build_context(dev_list, num, context_dict)
    test_feature_list = build_context(test_list, num, context_dict)

    return train_feature_list, dev_feature_list, test_feature_list
