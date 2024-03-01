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

def chinese_word_segmentation(text):
    '''按是否为汉字将文本分段'''
    pattern = re.compile(r'([\u4e00-\u9fa5]+|[^\u4e00-\u9fa5]+)')
    segments = pattern.findall(text)

    return segments

def text2pinyintoken(text):
    '''将文本中的汉字转化为拼音，将非汉字转化为`[OTHER]`'''
    pinyin = lazy_pinyin(text)
    pinyin_token = []
    i, j = 0, 0

    while j < len(text):
        if chinese_character(text[j]) or ling(text[j]):
            pinyin_token += [pinyin[i]]
            i += 1
            j += 1
        else:
            while j < len(text) and (not chinese_character(text[j])) and (not ling(text[j])):
                j += 1
                pinyin_token.append("[OTHER]")
            i += 1
    
    return pinyin_token

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

def replace_base_pinyin(text, confuse_data, similar_data, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
    '''
    将文本进行音似替换

    参数说明:
        p_confuse: 进行混淆替换的概率
        p_same: 混淆替换中，同音替换部分所占比例
        momentum: 减少连续替换出现的概率
    '''
    pinyin_token = text2pinyintoken(text)
    confused_text = ""
    confused_pinyin = []
    confused_label = []
    factor = 0
    for c, py in zip(text, pinyin_token):
        rand_a, rand_b = random.random(), random.random()
        if confuse_data.get(c, 0) != 0 and py != '[OTHER]' and rand_a < p_confuse*pow(momentum, factor):
            if rand_b < p_same or len(confuse_data[c]['similar_pronunciations']) == 0:
                confused_text += same_pinyin_char(c, py, confuse_data)
                confused_pinyin.append(py)
            else:
                char, pronun = similar_pinyin_char(c, py, confuse_data, similar_data)
                confused_text += char
                confused_pinyin.append(pronun)
        else:
            confused_text += c
            confused_pinyin.append(py)

        if confused_text[-1] != c:
            confused_label.append(1)
            factor += 1
        else:
            confused_label.append(0)
            factor = 0

    return confused_text, confused_pinyin, confused_label
