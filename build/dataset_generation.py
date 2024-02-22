"""用于针对不同任务生成不同数据集"""

from tqdm import tqdm
from utils import replace_base_pinyin

def generate_for_Detection(file, data, confuse_data, similar_data):
    with open(file, "a", encoding="utf-8") as f:
        for d in tqdm(data):
            confused_text, _, confused_label = replace_base_pinyin(d['correct_text'], confuse_data, similar_data)
            f.write(confused_text + "\t" + str(confused_label) + "\n")

def generate_for_PinyinMLM():
    pass

if __name__ == "__main__":
    import os
    import json
    dataset_dir = "../SIGHAN2015"
    with open(os.path.join(dataset_dir, "train.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open("../data/input/confused_dataset.json", 'r', encoding='utf-8') as f:
        confuse_data = json.load(f)

    with open("../data/input/similar_data.json", "r", encoding='utf-8') as f:
        similar_data = json.load(f)
    
    generate_for_Detection("../dataset/train.tsv", data, confuse_data, similar_data)
