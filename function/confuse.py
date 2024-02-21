import math
import random

from pinyin import *

# 参数
MAX_SAMPLE_NUM = 4  # 最大抽样数量
MAX_NGRAM_SAMPLE_NUM = 3
MAX_SHUFFLE_NUM = 4
MAX_DELETE_NUM = 3
SAMPLE_RATIO = 0.2  # 抽样比例
DELETE_SAMPLE_RATIO = 0.1
REPETE_SAMPLE_RATIO = 0.1
SHUFFLE_SAMPLE_RATIO = 0.15
SAME_TOPK = 20
SIMILAR_TOPK = 10


def load_confused_dataset(confused_file):
    confused_chinese_relation = {}
    with open(confused_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, value in data.items():
            similar_pronunciations = value['similar_pronunciations']
            same_pronunciation_chars = value['same_pronunciation_chars']
            same_pronunciation_chars = [item['char'] for item in same_pronunciation_chars]
            similar_pronunciation_chars = value['similar_pronunciation_chars']
            similar_pronunciation_chars = [item['char'] for item in similar_pronunciation_chars]
            similar_font_chars = value['similar_font_chars']
            similar_font_chars = [item['char'] for item in similar_font_chars]
            relation_items = {'similar_pronunciations':similar_pronunciations,
                              'same_pronunciation_chars': same_pronunciation_chars,
                              'similar_pronunciation_chars': similar_pronunciation_chars,
                              'similar_font_chars': similar_font_chars}
            confused_chinese_relation[key] = relation_items
    return confused_chinese_relation


# 读取汉字读音文件
pinyin_file = "../data/input/pinyin/汉字读音表gb2312.txt"
pinyin_dict = load_pinyin_to_dict(pinyin_file)
pinyin_tone_dict = load_pinyin_tone_to_dict(pinyin_file)
# 读取混淆集
confused_file = "../data/input/confused_dataset.json"
confused_dict = load_confused_dataset(confused_file)


def confuse_pinyin(text):
    # 2% 句子不替换
    text_replaced_prob = random.random()
    if text_replaced_prob < 0.02:
        replaced_text = text
    else:
        chars_num = len(text)
        sample_num = math.ceil(chars_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        sample_indices = random.sample(range(chars_num), selected_num)

        char_list = list(text)
        for idx in sample_indices:
            char = char_list[idx]
            pinyin = transform_chinese_to_pinyin(char)[0][0]
            if pinyin in pinyin_dict.keys():
                pinyin_prob = random.random()
                # 2/3概率用同音字替换
                if pinyin_prob <= 0.66:
                    same_pinyin_chars = pinyin_dict[pinyin]
                    if len(same_pinyin_chars) > 0:
                        # 给每个字符一个权重，前面的字符权重更大
                        weights = [1 / (i + 1) for i in range(len(same_pinyin_chars))]
                        # 根据权重选择一个字符
                        replace_char = random.choices(same_pinyin_chars, weights=weights, k=1)[0]
                        char_list[idx] = replace_char
                # 1/3概率用近音字替换
                else:
                    if char in confused_dict:
                        if confused_dict[char]['similar_pronunciations'] != []:
                            similar_pinyins = confused_dict[char]['similar_pronunciations']
                            similar_pinyin_chars = {}
                            for similar_pinyin in similar_pinyins:
                                similar_pinyin_chars[similar_pinyin] = pinyin_dict[similar_pinyin]
                            if len(similar_pinyin_chars) > 0:
                                # 随机选择要替换的近似拼音
                                replace_pinyin = random.choices(list(similar_pinyin_chars.keys()))[0]
                                replace_chars = similar_pinyin_chars[replace_pinyin]
                                # 给每个字符一个权重，前面的字符权重更大
                                weights = [1 / (i + 1) for i in range(len(replace_chars))]
                                # 根据权重选择一个字符
                                replace_char = random.choices(replace_chars, weights=weights, k=1)[0]
                                char_list[idx] = replace_char
        replaced_text = "".join(char_list)
    return replaced_text


def confuse_pinyin_tone(text):
    # 2% 句子不替换
    text_replaced_prob = random.random()
    if text_replaced_prob < 0.02:
        replaced_text = text
    else:
        chars_num = len(text)
        sample_num = math.ceil(chars_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        sample_indices = random.sample(range(chars_num), selected_num)

        char_list = list(text)
        for idx in sample_indices:
            char = char_list[idx]
            pinyin = transform_chinese_to_pinyin_tone(char)[0][0]
            if pinyin in pinyin_tone_dict.keys():
                pinyin_prob = random.random()
                # 2/3概率用同音字替换
                if pinyin_prob <= 0.66:
                    same_pinyin_chars = pinyin_tone_dict[pinyin]
                    if len(same_pinyin_chars) > 0:
                        # 给每个字符一个权重，前面的字符权重更大
                        weights = [1 / (i + 1) for i in range(len(same_pinyin_chars))]
                        # 根据权重选择一个字符
                        replace_char = random.choices(same_pinyin_chars, weights=weights, k=1)[0]
                        char_list[idx] = replace_char
                # 1/3概率用近音字替换
                # else:

        replaced_text = "".join(char_list)
    return replaced_text


if __name__ == '__main__':
    print(confuse_pinyin("啊"))