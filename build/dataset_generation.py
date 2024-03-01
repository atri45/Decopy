"""用于针对不同任务生成不同数据集"""

import os
from tqdm import tqdm
from utils import replace_base_pinyin, text2pinyintoken

def generate_for_Detection(file, data, confuse_data, similar_data, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            confused_text, _, confused_label = replace_base_pinyin(d['correct_text'], confuse_data, similar_data, p_confuse, p_same, momentum)
            f.write(confused_text + "\t" + str(confused_label) + "\n")

def generate_for_Correct(file, data, confuse_data, similar_data, pinyin_vocab, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            replaced_output = replace_base_pinyin(d['correct_text'], confuse_data, similar_data, p_confuse, p_same, momentum)
            pinyin_tokens = text2pinyintoken(replaced_output[0])
            # if '，' in pinyin_tokens:
            #     print(replaced_output[0], d['correct_text'])
            pinyin_ids = list(map(lambda x : pinyin_vocab[x], pinyin_tokens))
            f.write(replaced_output[0] + "\t" + d['correct_text'] + "\t" + str(pinyin_ids) + "\n")

def generate_for_PinyinMLM():
    pass

if __name__ == "__main__":
    import os
    import json
    dataset_dir = "SIGHAN2015"
    with open(os.path.join(dataset_dir, "train.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open("data/input/confused_dataset.json", 'r', encoding='utf-8') as f:
        confuse_data = json.load(f)

    with open("data/input/similar_data.json", "r", encoding='utf-8') as f:
        similar_data = json.load(f)

    pinyin_vocab = {}
    with open("build/pinyin_vocab.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            pinyin_vocab[line.strip("\n")] = i
    
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    generate_for_Correct("dataset/correct_train.tsv", data, confuse_data, similar_data, pinyin_vocab, momentum=1)