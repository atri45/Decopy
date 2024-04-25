"""废弃"""
import json
import os
from pycorrector.t5.t5_corrector import T5Corrector

def evaluate(file_path, model_dir, base_model="t5", batch_size=32):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    if base_model == "t5":
        corrector = T5Corrector(model_dir).batch_t5_correct

    original_sentences = [d["original_text"] for d in data]
    correct_sentences = [d["correct_text"] for d in data]
    predict_results = corrector(original_sentences, batch_size=batch_size)

    acc_num, total_num = 0, len(original_sentences)
    for i, result in enumerate(predict_results):
        if(result[0] == correct_sentences[i]):
            acc_num += 1

    accuracy = acc_num / total_num
    print(f"model: {model_dir.strip('/').split('/')[-1]}\n"
          f"dataset: {os.path.dirname(file_path).split('/')[-1]}\n"
          f"accuracy: {accuracy:.2f}")
