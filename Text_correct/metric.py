import numpy as np
from transformers import AutoTokenizer

pretrained_model = "Maciel/T5Corrector-base-v2"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

def computer_chinese_metrics(eval_preds):
    from rouge_chinese import Rouge
    rouge = Rouge()

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [list(pred) for pred in decoded_preds]    
    decoded_labels = [list(label) for label in decoded_labels]

    decoded_preds = [" ".join(pred_token) for pred_token in decoded_preds]
    decoded_labels = [" ".join(label_token) for label_token in decoded_labels]

    scores = rouge.get_scores(decoded_preds, decoded_labels)

    items = ['rouge-1', 'rouge-2', 'rouge-l']
    score = {}
    for item in items:
        item_score = []
        for score in scores:
            # print(score)
            item_score.append(score[item]['f'])
        mean_item_score = np.mean(item_score)
        score[item] = round(mean_item_score, 4)
    return score