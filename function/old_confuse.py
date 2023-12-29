import math
import random
from random import shuffle
import copy
import json
from LAC import LAC
from function.pronunciating import PronunciationRetrieval

# 参数
MAX_SAMPLE_NUM = 3
MAX_NGRAM_SAMPLE_NUM = 3
MAX_SHUFFLE_NUM = 4
MAX_DELETE_NUM = 3
SAMPLE_RATIO = 0.2
DELETE_SAMPLE_RATIO = 0.1
REPETE_SAMPLE_RATIO = 0.1
SHUFFLE_SAMPLE_RATIO = 0.15
SAME_TOPK = 20
SIMILAR_TOPK = 10
seg = LAC(mode='seg')


# 读取混淆集
def load_confused_dataset(confused_file):
    confused_chinese_relation = {}
    with open(confused_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, value in data.items():
            same_pronunciation_chars = value['same_pronunciation_chars']
            same_pronunciation_chars = [item['char'] for item in same_pronunciation_chars]
            similar_pronunciation_chars = value['similar_pronunciation_chars']
            similar_pronunciation_chars = [item['char'] for item in similar_pronunciation_chars]
            similar_font_chars = value['similar_font_chars']
            similar_font_chars = [item['char'] for item in similar_font_chars]
            relation_items = {'same_pronunciation_chars': same_pronunciation_chars,
                              'similar_pronunciation_chars': similar_pronunciation_chars,
                              'similar_font_chars': similar_font_chars}
            confused_chinese_relation[key] = relation_items
    return confused_chinese_relation


# 读取股票列表
def load_stock_list(stock_file):
    with open(stock_file, 'r') as f:
        stocks = [line.strip() for line in f.readlines()]
    return stocks


# 以字为单位进行干扰词替换
def replace_confused_char(text):
    # 2% 句子不替换
    text_replaced_prob = random.random()
    # print(text_replaced_prob)
    if text_replaced_prob < 0.02:
        replaced_text = text
    else:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        chars_num = len(text)
        sample_num = math.ceil(chars_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        sample_indices = random.sample(range(chars_num), selected_num)
        # print(sample_indices)

        words_list = list(text)
        for idx in sample_indices:
            char = words_list[idx]
            if char in confused_chinese_relation:
                prob = random.random()
                # 1/3的字使用同音字替换
                if prob <= 0.33:
                    same_pronunciation_chars = confused_chinese_relation[char]['same_pronunciation_chars']
                    if len(same_pronunciation_chars) > 0:
                        replaced_char = random.choice(same_pronunciation_chars)
                        words_list[idx] = replaced_char
                # 1/3的字使用近音字替换
                elif prob <= 0.66:
                    similar_pronunciation_chars = confused_chinese_relation[char]['similar_pronunciation_chars']
                    if len(similar_pronunciation_chars) > 0:
                        replaced_char = random.choice(similar_pronunciation_chars)
                        words_list[idx] = replaced_char
                # 1/3的字使用近形字替换
                else:
                    similar_font_chars = confused_chinese_relation[char]['similar_font_chars']
                    if len(similar_font_chars) > 0:
                        replaced_char = random.choice(similar_font_chars)
                        words_list[idx] = replaced_char

        replaced_text = "".join(words_list)
    return replaced_text


# 以词为单位进行干扰词替换
def replace_confused_word(text):
    # 分词
    words = seg.run(text)
    ngram_word_locs = [idx for idx, word in enumerate(words) if len(word) > 1]

    if len(ngram_word_locs) > 0:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        ngram_words_num = len(ngram_word_locs)
        sample_num = math.ceil(ngram_words_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_NGRAM_SAMPLE_NUM)
        sample_indices = random.sample(range(ngram_words_num), selected_num)

        ngram_loc = [ngram_word_locs[index] for index in sample_indices]
        for loc in ngram_loc:
            word = words[loc]
            prob = random.random()
            if prob <= 0.8:
                # 4/5的字使用同音字替换
                pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word)
                same_pronunciation_words = pronunciation_retrieval.get_same_pronunciation_word(pronunciations,
                                                                                               SAME_TOPK)
                same_pronunciation_words = pronunciation_retrieval.filter_same_word(same_pronunciation_words, word)
                if len(same_pronunciation_words) > 0:
                    replace_word = random.choice(same_pronunciation_words)
                else:
                    replace_word = word
            else:
                # 1/5的字使用近音字替换
                pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word)
                similar_pronunciations = pronunciation_retrieval.get_similar_pronunciations(pronunciations)
                similar_pronunciation_words = pronunciation_retrieval.get_similar_pronunciation_words(
                    similar_pronunciations, SIMILAR_TOPK)
                if len(similar_pronunciation_words) > 0:
                    replace_word = random.choice(similar_pronunciation_words)
                else:
                    replace_word = word
            words[loc] = replace_word
        replace_text = "".join(words)
        return replace_text
    else:
        return None


# 删除字
def delete_char(text):
    # 10% 句子不做处理
    if random.random() < 0.1:
        replaced_text = text
    else:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        words_list = list(text)
        chars_num = len(text)
        sample_num = math.ceil(chars_num * DELETE_SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        selected_indices = random.sample(range(chars_num), selected_num)
        # print(sample_indices)
        # 需要对要删除的索引进行排序，以便从后往前删除，避免索引错位
        selected_indices.sort(reverse=True)

        for idx in selected_indices:
            words_list.pop(idx)

        replaced_text = "".join(words_list)

    return replaced_text


# 删除词(纠错难度太高了，更类似于根据句意填补文本)
def delete_word(text):
    # 10% 句子不做处理
    if random.random() < 0.1:
        replaced_text = text
    else:
        # 分词
        words = seg.run(text)
        word_num = len([word for word in words if len(word) > 1])
        deleted_num = math.ceil(word_num * DELETE_SAMPLE_RATIO)
        selected_num = min(deleted_num, MAX_DELETE_NUM)

        ngram_word_indices = [wid for wid, word in enumerate(words) if len(word) > 1]
        selected_indices = random.sample(ngram_word_indices, selected_num)
        # 需要对要删除的索引进行排序，以便从后往前删除，避免索引错位
        selected_indices.sort(reverse=True)

        replaced_words = copy.deepcopy(words)
        for index in selected_indices:
            replaced_words.pop(index)

        replaced_text = "".join(replaced_words)

    return replaced_text


# 重复增加字（冗余）
def repeat_char(text):
    # 10% 句子不做处理
    if random.random() < 0.1:
        replaced_text = text
    else:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        chars_list = list(text)
        chars_num = len(text)
        sample_num = math.ceil(chars_num * REPETE_SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        sample_indices = random.sample(range(chars_num), selected_num)

        for idx in sample_indices:
            char_to_repeat = chars_list[idx]
            chars_list[idx] = char_to_repeat + char_to_repeat  # 重复一次

        replaced_text = "".join(chars_list)

    return replaced_text


# 重复增加词（冗余）
def repeat_word(text):
    # 10% 句子不做处理
    if random.random() < 0.1:
        replaced_text = text
    else:
        # 分词
        word_list = seg.run(text)
        word_num = len([word for word in word_list if len(word) > 1])
        deleted_num = math.ceil(word_num * DELETE_SAMPLE_RATIO)
        selected_num = min(deleted_num, MAX_DELETE_NUM)

        ngram_word_indices = [wid for wid, word in enumerate(word_list) if len(word) > 1]
        selected_indices = random.sample(ngram_word_indices, selected_num)

        for idx in selected_indices:
            word_to_repeat = word_list[idx]
            word_list[idx] = word_to_repeat + word_to_repeat  # 重复一次

        replaced_text = "".join(word_list)

    return replaced_text


# 混拼音
def replace_word_with_pronunciation(text):
    # 10% 句子不做处理
    if random.random() < 0.1:
        replaced_text = text
    else:
        # 分词
        word_list = seg.run(text)
        word_num = len([word for word in word_list if len(word) > 1])
        deleted_num = math.ceil(word_num * DELETE_SAMPLE_RATIO)
        selected_num = min(deleted_num, MAX_DELETE_NUM)

        ngram_word_indices = [wid for wid, word in enumerate(word_list) if len(word) > 1]
        selected_indices = random.sample(ngram_word_indices, selected_num)

        for idx in selected_indices:
            word_to_replace = word_list[idx]
            pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word_to_replace)
            word_list[idx] = "".join(pronunciations)

        replaced_text = "".join(word_list)

    return replaced_text


# 打乱词序
def shuffle_word(text):
    words = seg.run(text)
    # 随机打乱ngram的词序
    word_num = len([word for word in words if len(word) > 1])
    shuffle_num = math.ceil(word_num * SHUFFLE_SAMPLE_RATIO)
    selected_num = min(shuffle_num, MAX_SHUFFLE_NUM)

    ngram_word_indices = [wid for wid, word in enumerate(words) if len(word) > 1]
    selected_indices = random.sample(ngram_word_indices, selected_num)

    replace_words = copy.deepcopy(words)
    for index in selected_indices:
        selected_word = replace_words[index]
        chars_list = list(selected_word)

        if len(chars_list) == 2:
            chars_list = [chars_list[1], chars_list[0]]
        else:
            shuffle(chars_list)
        replace_word = "".join(chars_list)
        replace_words[index] = replace_word
    replace_text = "".join(replace_words)
    return replace_text


if __name__ == '__main__':
    # 加载同音字和近音字检索模块
    common_file = "../data/input/chinese_3500.txt"
    chinese_pronunciation_file = "../data/input/chinese_pronunciation.txt"
    pronunciation_retrieval = PronunciationRetrieval(common_file, chinese_pronunciation_file)

    # 读取混淆集
    confused_file = "../data/input/confused_dataset.json"
    confused_chinese_relation = load_confused_dataset(confused_file)

    # 股票名称作为专业名称进行分词，加入固定此表中
    stock_file = "../data/input/stock.txt"
    seg.load_customization(stock_file, sep=None)

    print("成功读取输入文件。")