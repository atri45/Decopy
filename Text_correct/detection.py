'''用于检测部分的推理'''
import torch
import sys
sys.path.append("..")

from build.utils import text2token

def detect(model, tokenizer, text, threshold_p, show_error=False):
    device = model.device
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    _, pinyin_tokens = text2token(text[0], tokenizer)

    outputs = model(**inputs).logits
    outputs = torch.nn.functional.softmax(outputs, dim=-1)[:, :, 1]
    norm_outputs = outputs.clone()
    for i, _ in enumerate(pinyin_tokens):
        if i == 0 or pinyin_tokens[i-1] == '[OTHER]':
            norm_outputs[0][i+1] = 0.6*outputs[0][i+1] + 0.4*outputs[0][i+2]
        elif i == len(pinyin_tokens)-1 or pinyin_tokens[i+1] == '[OTHER]':
            norm_outputs[0][i+1] = 0.6*outputs[0][i+1] + 0.4*outputs[0][i]
        else:
            norm_outputs[0][i+1] = 0.5*outputs[0][i+1] + 0.25*outputs[0][i+2] + 0.25*outputs[0][i]

    if show_error:
        print("error:", end="")
        for i, l in enumerate(norm_outputs[0][1:-1]):
            print(f"{text[0][i]}: {norm_outputs[0][i+1]:.3f}")

    outputs = (outputs > threshold_p).int()
    norm_outputs = (norm_outputs > threshold_p).int()
    return outputs, norm_outputs


if __name__ == "__main__":
    from transformers import BertForTokenClassification, BertTokenizer
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained("pretrained_models/bert-base-chinese", use_safetensors=True)
    tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
    state_dict = torch.load("check_point/bert_base_cn_detect_0.ckpt")
    model.load_state_dict(state_dict)
    model.eval()

    text = ["我的浮木都在国企上班", "任何不位法的行为偷是合法的", "我怀念的，是五话不说", "你怎么这么户秃"]
    output = detect(model, tokenizer, text, 0.1, True)
