import sys
sys.path.append("..")
sys.path.append("Text_correct")

import torch
import json
from build.utils import text2token, token2ids
from transformers import BertTokenizer, BertForTokenClassification, BertConfig
from model import PinyinBertForMaskedLM
from detection import detect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("Decbert")
with open("Decbert/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
config = BertConfig(**config)

detect_model = BertForTokenClassification(config=config).to(device)
correct_model = PinyinBertForMaskedLM(config=config).to(device)

detect_state_dict = torch.load("Decbert/detect_model.bin", map_location=device)
correct_state_dict = torch.load("Decbert/correct_model.bin", map_location=device)
detect_model.load_state_dict(detect_state_dict)
correct_model.load_state_dict(correct_state_dict)

detect_model.eval()
correct_model.eval()
pinyin_vocab = {}

with open("Decbert/pinyin_vocab.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        pinyin_vocab[line.strip("\n")] = i

def detect(model, tokenizer, text, threshold_p, show_error=False):
    device = model.device
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    outputs = model(**inputs).logits
    outputs = torch.nn.functional.softmax(outputs, dim=-1)[:, :, 1]
    norm_outputs = outputs.clone() # 废弃

    if show_error:
        print("error:", end="")
        for i, l in enumerate(norm_outputs[0][1:-1]):
            print(f"{text[0][i]}: {norm_outputs[0][i+1]:.3f}")

    outputs = (outputs > threshold_p).int()
    norm_outputs = (norm_outputs > threshold_p).int()
    return outputs, norm_outputs

def predict(text, threshold_p=0.67):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    error_label = detect(detect_model, tokenizer, [text], threshold_p, show_error=False)[0].unsqueeze(-1).to(device)
    _, pinyin_token = text2token(text, tokenizer)
    pinyin_ids = torch.tensor([0] + token2ids(pinyin_token, pinyin_vocab) + [0], device=device)
    output = correct_model(**inputs, pinyin_ids=pinyin_ids, error_prob=error_label)

    predict_ids_for_output = output[0].argmax(-1)[0]
    result = "".join(tokenizer.convert_ids_to_tokens(predict_ids_for_output)[1:-1])
    return result

if __name__ == "__main__":
    text = "我喜欢泡步，田野在我面前倒戴，云彩也没我跑得快"
    print(predict(text))
