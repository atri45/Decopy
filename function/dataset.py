import csv
from zhconv import convert

from function.confuse import confuse_pinyin
from function.old_confuse import *


# 获取不一致字词的位置
def get_different_id(original_text, replaced_text):
    wrong_ids = []

    # 以最短的字符串长度为限制进行比较
    min_length = min(len(original_text), len(replaced_text))
    for i in range(min_length):
        if original_text[i] != replaced_text[i]:
            wrong_ids.append(i)

    # 如果一个文本比另一个文本长，则较长文本中的所有其他字符位置都被认为是不同的
    for i in range(min_length, max(len(original_text), len(replaced_text))):
        wrong_ids.append(i)

    # 将wrong_ids转化为特定格式的字符串
    return ', '.join(map(str, wrong_ids))


# 生成用mask替换错误位置的数据集
def mask_dataset(dataset_path, output_path):
    # 读取文件内容
    with open(dataset_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 处理每个字典
    for entry in data:
        if entry['wrong_ids']:
            entry['original_text'] = entry['replaced_text']
            replaced_text = list(entry['replaced_text'])
            wrong_ids = map(int, entry['wrong_ids'].split(', '))
            # 替换原始文本中的错误位置为[MASK]
            for wrong_id in wrong_ids:
                replaced_text[wrong_id] = '[MASK]'
            # 将替换后的文本更新到字典中
            entry['replaced_text'] = ''.join(replaced_text)

    # 将更新后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


# 读取数据集，为文本增加混淆，并保存结果数据集（按需求更改内部混淆逻辑）
def confuse_dataset_from_json(function, dataset_path, output_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for example in dataset:
        text = example["original_text"]
        # 针对部分繁体字，先做文字简写
        text = convert(text, 'zh-cn')

        # 可以设置一条记录生成几条记录
        for _ in range(1):
            source = function(text)
            result_entry = {
                "replaced_text": source,
                "wrong_ids": get_different_id(text, source),
                "original_text": text
            }
            results.append(result_entry)

    # 将结果保存为 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# 读取数据集，为文本增加混淆，并保存结果数据集（按需求更改内部混淆逻辑）
def confuse_dataset_from_tsv(function, dataset_path, output_path):
    with open(dataset_path, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        results = []
        for row in tsvreader:
            text = row[1]
            # 针对部分繁体字，先做文字简写
            text = convert(text, 'zh-cn')

            # 可以设置一条记录生成几条记录
            for _ in range(1):
                source = function(text)
                result_entry = {
                    "replaced_text": source,
                    "wrong_ids": get_different_id(text, source),
                    "original_text": text
                }
                results.append(result_entry)

    # 将结果保存为 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 调用函数处理数据集并保存结果集
    confuse_dataset_from_json(confuse_pinyin,
                    "../data/input/train.json",
                              "../data/input/pinyin/confused_train.json")
    # mask_dataset("../data/input/pinyin/confused_train.json",
    #              "../data/input/pinyin/masked_train.json")
    print("数据集处理完成并已保存。")
