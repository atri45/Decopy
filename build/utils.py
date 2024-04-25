"""其他工具"""

import re
import random
from tqdm import tqdm
from pypinyin import lazy_pinyin

def chinese_character(c):
    '''判断一个字符是否为汉字'''
    return '\u4e00' <= c <= '\u9fa5'

def ling(c):
    '''判断一个字符是否为汉字"〇"'''
    return c == '\u3007'

def get_chinese_part(text):
    '''得到文本中的中文部分'''
    pattern = re.compile(r'([\u3007\u4e00-\u9fa5]+)')
    segments = pattern.findall(text)

    return "".join(segments)

def text2token(text, tokenizer):
    '''将文本中的汉字转化为拼音，将非汉字转化为`[OTHER]`'''
    cn_char_pinyin = lazy_pinyin(get_chinese_part(text))
    text_tokens = tokenizer.tokenize(text)
    text_tokens_no_unk = []
    for i in text_tokens:
        if i != "[UNK]":
            text_tokens_no_unk.append(i)
            text = text[len(i.strip("##")):]
        else:
            text_tokens_no_unk.append(text[0])
            text = text[1:]
    pinyin_tokens = []
    i = 0
    for token in text_tokens_no_unk:
        if chinese_character(token) or ling(token):
            pinyin_tokens.append(cn_char_pinyin[i])
            i += 1
        else:
            pinyin_tokens.append("[OTHER]")
    return text_tokens_no_unk, pinyin_tokens

def token2ids(pinyin_tokens, pinyin_vocab):
    '''根据tokenizer的分词规则，将中文部分转化为pinyin_vocab中的id，非中文部分转化为0'''
    pinyin_ids = list(map(lambda x: pinyin_vocab.get(x, 0), pinyin_tokens))
    return pinyin_ids

def same_pinyin_char(c, pronun, confuse_data):
    '''获取同音字'''
    same_pinyin_chars = [i['char'] for i in confuse_data[c]['same_pronunciation_chars'] if i['pronunciation'] == pronun]

    if len(same_pinyin_chars) == 0:
        return c

    confuse_char = random.sample(same_pinyin_chars, 1)[0]
    return confuse_char

def similar_pinyin_char(c, pronun, confuse_data, similar_data):
    '''获取音似字，如果没有则返回其本身'''
    similar_pinyin_chars = [i for i in confuse_data[c]['similar_pronunciation_chars'] if i["pronunciation"] in similar_data[pronun]]

    if len(similar_pinyin_chars) == 0:
        return c, pronun

    similar_result = random.sample(similar_pinyin_chars, 1)[0]
    return similar_result['char'], similar_result['pronunciation']

def visualize_download(resp, write_file, proc_name, total, block_size=1024):
    '''可视化文件下载过程'''
    with open(write_file, 'wb') as file:
        progress_bar = tqdm(total=total, desc=proc_name, unit='iB', unit_scale=True, unit_divisor=1024)
        for data in resp.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
        progress_bar.close()

def replace_base_pinyin(text, confuse_data, similar_data, tokenizer, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    '''
    将文本进行音似替换

    参数说明:
        p_confuse: 进行混淆替换的概率
        p_same: 混淆替换中，同音替换部分所占比例
        momentum: 减少连续替换出现的概率
    '''
    text_tokens, pinyin_tokens = text2token(text, tokenizer)
    confused_tokens = []
    confused_pinyin = []
    confused_label = []
    factor = 0
    for tx, py in zip(text_tokens, pinyin_tokens):
        rand_a, rand_b = random.random(), random.random()
        if confuse_data.get(tx, 0) != 0 and py != '[OTHER]' and rand_a < p_confuse*pow(momentum, factor):
            if rand_b < p_same or len(confuse_data[tx]['similar_pronunciations']) == 0:
                confused_tokens.append(same_pinyin_char(tx, py, confuse_data))
                confused_pinyin.append(py)
            else:
                char, pronun = similar_pinyin_char(tx, py, confuse_data, similar_data)
                confused_tokens.append(char)
                confused_pinyin.append(pronun)
        else:
            confused_tokens.append(tx)
            confused_pinyin.append(py)

        if confused_tokens[-1] != tx:
            confused_label.append(1)
            factor += 1
        else:
            confused_label.append(0)
            factor = 0
    confused_text = "".join(confused_tokens).replace("##", "")
    return confused_text, confused_pinyin, confused_label

def same_pinyin_char_1(c, pronun, confuse_data):
    '''获取同音字'''
    same_pinyin_chars = [i for i in confuse_data[pronun]['same_pronunciation_chars'] if i != c]

    if len(same_pinyin_chars) == 0:
        return c

    confuse_char = random.sample(same_pinyin_chars, 1)[0]
    return confuse_char

def similar_pinyin_char_1(c, pronun, confuse_data):
    '''获取音似字，如果没有则返回其本身'''
    similar_pinyins = [i for i in confuse_data[pronun]['similar_pronunciations'] if i != pronun]

    if len(similar_pinyins) == 0:
        return c, pronun

    similar_pinyin = random.sample(similar_pinyins, 1)[0]
    similar_pinyin_chars= [i for i in confuse_data[similar_pinyin]['same_pronunciation_chars']]
    similar_result = random.sample(similar_pinyin_chars, 1)[0], similar_pinyin
    return similar_result

def replace_base_pinyin_1(text, confuse_data, tokenizer, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    '''
    将文本进行音似替换

    参数说明:
        p_confuse: 进行混淆替换的概率
        p_same: 混淆替换中，同音替换部分所占比例
        momentum: 减少连续替换出现的概率
    '''
    text_tokens, pinyin_tokens = text2token(text, tokenizer)
    confused_tokens = []
    confused_pinyin = []
    confused_label = []
    factor = 0
    for tx, py in zip(text_tokens, pinyin_tokens):
        rand_a, rand_b = random.random(), random.random()
        if py != '[OTHER]' and rand_a < p_confuse*pow(momentum, factor):
            if rand_b < p_same or len(confuse_data[py]['similar_pronunciations']) == 0:
                confused_tokens.append(same_pinyin_char_1(tx, py, confuse_data))
                confused_pinyin.append(py)
            else:
                char, pronun = similar_pinyin_char_1(tx, py, confuse_data)
                confused_tokens.append(char)
                confused_pinyin.append(pronun)
        else:
            confused_tokens.append(tx)
            confused_pinyin.append(py)

        if confused_tokens[-1] != tx:
            confused_label.append(1)
            factor += 1
        else:
            confused_label.append(0)
            factor = 0
    confused_text = "".join(confused_tokens).replace("##", "")
    return confused_text, confused_pinyin, confused_label

def transfer_sighan_item(d, tokenizer):
    original_tokens = text2token(d["original_text"], tokenizer)
    correct_tokens = text2token(d["correct_text"], tokenizer)
    confused_label = []
    confused_pinyin = original_tokens[1]
    for o_t, c_t in zip(original_tokens[0], correct_tokens[0]):
        if o_t != c_t:
            confused_label.append(1)
        else:
            confused_label.append(0)
    return d["original_text"], confused_pinyin, confused_label

