import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='./data/Tencent_AILab_ChineseEmbedding.txt')
args = parser.parse_args()

file_path = args.file_path

ret_list = []

with open(file_path, "r") as f:
    while True:
        item = f.readline()
        if not item:
            break
        items = item.split()
        # only choose unigram
        if len(items) > 2 and len(items[0]) == 1:
            ret_list.append(item)

with open("./data/tencent_unigram.txt", "w+", encoding="utf-8") as f:
    f.write("".join(ret_list))
