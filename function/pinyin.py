from pypinyin import pinyin, Style
import json

# 计数文本错误中各拼音类型的数量
def count_pinyin_errors(filePath):
    with open(filePath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dict = {}
    for example in dataset:
        original_text = example["original_text"]
        correct_text = example["correct_text"]
        wrong_ids = example["wrong_ids"]
        for i in wrong_ids:
            correct_py = transform_chinese_to_pinyin(correct_text[i])
            original_py = transform_chinese_to_pinyin(original_text[i])
            keys = dict.keys()
            if correct_py[0][0]+correct_py[0][1] != original_py[0][0]+original_py[0][1]:
                if correct_py[0][0]+correct_py[0][1]+"<>"+original_py[0][0]+original_py[0][1] not in keys:
                    dict[correct_py[0][0]+correct_py[0][1]+"<>"+original_py[0][0]+original_py[0][1]] = 0
                dict[correct_py[0][0]+correct_py[0][1]+"<>"+original_py[0][0]+original_py[0][1]] += 1
            # 替换等式左边字符串可以打印查看错误的字是啥
            if '' == correct_py[0][0] + correct_py[0][1] + "<>" + original_py[0][0] + original_py[0][1]:
                print(correct_text[i]+"->"+original_text[i])
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print("-------------拼音错误-------------")
    for i in dict:
        print(i)


# 计数文本错误中各声母类型的数量
def count_shengmu_errors(filePath):
    with open(filePath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dict = {}
    for example in dataset:
        original_text = example["original_text"]
        correct_text = example["correct_text"]
        wrong_ids = example["wrong_ids"]
        for i in wrong_ids:
            correct_py = transform_chinese_to_pinyin(correct_text[i])
            original_py = transform_chinese_to_pinyin(original_text[i])
            keys = dict.keys()
            if correct_py[0][0] != original_py[0][0]:
                if correct_py[0][0]+"<>"+original_py[0][0] not in keys:
                    dict[correct_py[0][0] + "<>" + original_py[0][0]] = 0
                dict[correct_py[0][0]+"<>"+original_py[0][0]] += 1
            # 替换等式左边字符串可以打印查看错误的字是啥
            if '' == correct_py[0][0] + "<>" + original_py[0][0]:
                print(correct_text[i]+"->"+original_text[i])
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print("-------------声母错误-------------")
    for i in dict:
        print(i)


# 计数文本错误中各韵母类型的数量
def count_yunmu_errors(filePath):
    with open(filePath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dict = {}
    for example in dataset:
        original_text = example["original_text"]
        correct_text = example["correct_text"]
        wrong_ids = example["wrong_ids"]
        for i in wrong_ids:
            correct_py = transform_chinese_to_pinyin(correct_text[i])
            original_py = transform_chinese_to_pinyin(original_text[i])
            keys = dict.keys()
            if correct_py[0][1] != original_py[0][1]:
                if correct_py[0][1]+"<>"+original_py[0][1] not in keys:
                    dict[correct_py[0][1] + "<>" + original_py[0][1]] = 0
                dict[correct_py[0][1]+"<>"+original_py[0][1]] += 1
            # 替换等式左边字符串可以打印查看错误的字是啥
            if '' == correct_py[0][1] + "<>" + original_py[0][1]:
                print(correct_text[i]+"->"+original_text[i])
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print("-------------韵母错误-------------")
    for i in dict:
        print(i)


# 将中文转为拼音
def transform_chinese_to_pinyin(text):
    # 使用Style.INITIALS风格获取声母列表
    initials = pinyin(text, style=Style.INITIALS, strict=False, neutral_tone_with_five=True, errors=lambda x: [['null'] for _ in x])
    # 使用Style.FINALS_TONE3风格获取韵母列表
    finals_tone3 = pinyin(text, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, errors=lambda x: [['nulll'] for _ in x])

    result = []
    for i in range(len(initials)):
        initial = initials[i][0]  # 获取声母
        final = finals_tone3[i][0][:-1]  # 获取韵母
        py = initial+final
        # 匹配汉字读音表gb2312
        if py == 'lve':
            py = 'lue'
        if py == 'nve':
            py = 'nue'
        result.append([py])

    return result


# 将中文转为拼音+音调
def transform_chinese_to_pinyin_tone(text):
    # 使用Style.INITIALS风格获取声母列表
    initials = pinyin(text, style=Style.INITIALS, strict=False, neutral_tone_with_five=True, errors=lambda x: [['null'] for _ in x])
    # 使用Style.FINALS_TONE3风格获取韵母列表
    finals_tone3 = pinyin(text, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, errors=lambda x: [['nulll'] for _ in x])

    result = []
    for i in range(len(initials)):
        initial = initials[i][0]  # 获取声母
        final = finals_tone3[i][0]#[:-1]  # 获取韵母
        result.append([initial+final])

    return result


# 将中文转为声母+韵母
def transform_chinese_to_shengmu_yunmu(text):
    # 使用Style.INITIALS风格获取声母列表
    initials = pinyin(text, style=Style.INITIALS, strict=False, neutral_tone_with_five=True, errors=lambda x: [['null'] for _ in x])
    # 使用Style.FINALS_TONE3风格获取韵母列表
    finals_tone3 = pinyin(text, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, errors=lambda x: [['nulll'] for _ in x])

    result = []
    for i in range(len(initials)):
        initial = initials[i][0]  # 获取声母
        final = finals_tone3[i][0][:-1]  # 获取韵母
        result.append([initial, final])

    return result


# 将中文转为声母+韵母+音调
def transform_chinese_to_shengmu_yunmu_tone(text):
    # 使用Style.INITIALS风格获取声母列表
    initials = pinyin(text, style=Style.INITIALS, strict=False, neutral_tone_with_five=True, errors=lambda x: [['null'] for _ in x])
    # 使用Style.FINALS_TONE3风格获取韵母列表
    finals_tone3 = pinyin(text, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True, errors=lambda x: [['nulll'] for _ in x])

    result = []
    for i in range(len(initials)):
        initial = initials[i][0]  # 获取声母
        final = finals_tone3[i][0]  # 获取韵母
        result.append([initial, final])

    return result


# 加载拼音+音调到字典中
def load_pinyin_tone_to_dict(filename):
    pinyin_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割读音和对应的汉字
            parts = line.strip().split(':')
            if len(parts) != 2:
                continue
            pinyin, characters = parts
            # 将汉字添加到字典中对应的读音里
            pinyin_dict.setdefault(pinyin, []).extend(characters)

    return pinyin_dict


# 加载拼音到字典中
def load_pinyin_to_dict(filename):
    pinyin_dict = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) != 2:
                continue
            pinyin, characters = parts
            # 忽略读音的声调
            pinyin_key = ''.join(filter(str.isalpha, pinyin))
            if pinyin_key not in pinyin_dict:
                pinyin_dict[pinyin_key] = []
            # 追加当前声调的汉字列表
            pinyin_dict[pinyin_key].append(characters)
    # 重新组织每个拼音下的汉字列表
    for key in pinyin_dict:
        combined_list = []
        lists = pinyin_dict[key]
        max_length = max(len(lst) for lst in lists)
        for i in range(max_length):
            for lst in lists:
                if i < len(lst):
                    combined_list.append(lst[i])
        pinyin_dict[key] = combined_list

    return pinyin_dict


if __name__ == '__main__':
    text = "基于深度学习的中文文本纠错系统。"
    print(transform_chinese_to_pinyin(text))

    # count_pinyin_errors("../data/input/test.json")
    # count_shengmu_errors("../data/input/test.json")
    # count_yunmu_errors("../data/input/test.json")