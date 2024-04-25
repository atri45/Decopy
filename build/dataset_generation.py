"""用于针对不同任务生成不同数据集"""

import os
from tqdm import tqdm
from utils import replace_base_pinyin_1, text2token, token2ids, replace_base_pinyin, transfer_sighan_item

def generate_for_Detection(file, data, confuse_data, similar_data, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            confused_text, _, confused_label = replace_base_pinyin(d['correct_text'], confuse_data, similar_data, p_confuse, p_same, momentum)
            f.write(confused_text + "\t" + str(confused_label) + "\n")

def generate_for_all(file, data, confuse_data, pinyin_vocab, tokenizer, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            replaced_output = replace_base_pinyin_1(line, confuse_data, tokenizer, p_confuse, p_same, momentum)
            pinyin_ids = token2ids(replaced_output[1], pinyin_vocab)
            f.write(replaced_output[0] + "\t" + d['correct_text'] + "\t" + str(replaced_output[2]) + "\t" + str(pinyin_ids) + "\n")

def generate_for_all_2kk(file, data, confuse_data, pinyin_vocab, tokenizer, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    with open(file, "a", encoding="utf-8") as f:
        for line in tqdm(data):
            line = line.strip("\n")
            replaced_output = replace_base_pinyin_1(line, confuse_data, tokenizer, p_confuse, p_same, momentum)
            pinyin_ids = token2ids(replaced_output[1], pinyin_vocab)
            f.write(replaced_output[0] + "\t" + line + "\t" + str(replaced_output[2]) + "\t" + str(pinyin_ids) + "\n")

def generate_for_sighan(file, data, tokenizer, pinyin_vocab):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            norm_output = transfer_sighan_item(d, tokenizer)
            pinyin_ids = token2ids(norm_output[1], pinyin_vocab)
            f.write(norm_output[0] + "\t" + d['correct_text'] + "\t" + str(norm_output[2]) + "\t" + str(pinyin_ids) + "\n")
    

if __name__ == "__main__":
    import os
    import json
    from transformers import BertTokenizer
    dataset_dir = "SIGHAN2015"
    with open(os.path.join(dataset_dir, "train.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
    # with open("pre_train_dataset/pre_traindata.txt", "r", encoding='utf-8') as f:
    #     data = f.readlines()

    # with open("data/input/confusion.json", 'r', encoding='utf-8') as f:
    #     confuse_data = json.load(f)

    # with open("data/input/similar_data.json", "r", encoding='utf-8') as f:
    #     similar_data = json.load(f)

    pinyin_vocab = {}
    with open("build/pinyin_vocab.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            pinyin_vocab[line.strip("\n")] = i
    tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")

    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    # generate_for_all_2kk("dataset/all_train_4.tsv", data, confuse_data, pinyin_vocab, tokenizer ,momentum=1)
    generate_for_sighan("data/original_sighan_train.tsv", data, tokenizer, pinyin_vocab)
    # text = "快把cpucpu还给我"
    # replaced_output = replace_base_pinyin(text, confuse_data, similar_data, tokenizer)
    # pinyin_ids = token2ids(replaced_output[1], pinyin_vocab)
    # print(replaced_output[0] + "\t" + text + "\t" + str(replaced_output[2]) + "\t" + str(pinyin_ids) + "\n")
