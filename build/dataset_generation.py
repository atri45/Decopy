"""用于针对不同任务生成不同数据集"""

from utils import *

def replace_base_pinyin(text, confuse_data, p_confuse: float=0.1, p_same: float=0.6, momentum: float=0.9):
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
        if py != '[OTHER]' and rand_a < p_confuse*pow(momentum, factor):
            if rand_b < p_same or len(confuse_data[c]['similar_pronunciations']) == 0:
                confused_text += same_pinyin_char(c, py, confuse_data)
                confused_pinyin.append(py)
            else:
                char, pronun = similar_pinyin_char(c, py, confuse_data)
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

def generate_for_Detection(dataset_file):
    pass

def generate_for_PinyinMLM():
    pass

