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
        result.append([initial, final])

    return result

if __name__ == '__main__':

    text = "基于深度学习的中文文本纠错系统。"
    print(transform_chinese_to_pinyin(text))

    # count_pinyin_errors("../data/input/test.json")
    count_shengmu_errors("../data/input/test.json")
    count_yunmu_errors("../data/input/test.json")