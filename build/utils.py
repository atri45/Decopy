"""其他工具"""

import re
import random
from tqdm import tqdm
from pypinyin import lazy_pinyin

def chinese_character(c):
    '''判断一个字符是否为汉字'''
    return '\u4e00' <= c <= '\u9fa5'

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
        if chinese_character(text[j]):
            pinyin_token += [pinyin[i]]
            i += 1
            j += 1
        else:
            while not chinese_character(text[j]):
                j += 1
                pinyin_token.append("[OTHER]")
            i += 1
    
    return pinyin_token

def same_pinyin_char(c, pronun, confuse_data):
    '''获取同音字'''
    same_pinyin_chars = [i['char'] for i in confuse_data[c]['same_pronunciation_chars'] if i['pronuncation'] == pronun]
    while True:
        confuse_char = random.sample(same_pinyin_chars, 1)[0]
        if len(confuse_data[confuse_char]['pronunciations']) == 1:
            return confuse_char

def similar_pinyin_char(c, pronun, confuse_data):
    '''获取音似字，如果没有则返回其本身'''
    if len(confuse_data[c]['pronunciations']) == 1:
        similar_pinyin_chars = [i['char'] for i in confuse_data[c]['similar_pronunciation_chars']]
    else:
        brige_c = same_pinyin_char(c, pronun, confuse_data)
        similar_pinyin_chars = [i['char'] for i in confuse_data[brige_c]['similar_pronunciation_chars']]
    
    if len(similar_pinyin_chars) == 0:
        return c

    while True:
        confuse_char = random.sample(similar_pinyin_chars, 1)[0]
        if len(confuse_data[confuse_char]['pronunciations']) == 1 and confuse_char != c:
            return confuse_char, confuse_data[confuse_char]['pronunciations'][0]

def visualize_download(resp, write_file, proc_name, total, block_size=1024):
    '''可视化文件下载过程'''
    with open(write_file, 'wb') as file:
        progress_bar = tqdm(total=total, desc=proc_name, unit='iB', unit_scale=True, unit_divisor=1024)
        for data in resp.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
        progress_bar.close()