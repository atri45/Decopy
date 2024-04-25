import torch
from tqdm import tqdm
from torch import nn
from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import DataLoader
from CSCDatasets import CSCDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("pretrained_models/bert-base-chinese")
# corrector = PinyinBertForMaskedLM.from_pretrained("pretrained_models/bert-base-chinese",  use_safetensors=True).to(device)
detection = BertForTokenClassification.from_pretrained("pretrained_models/bert-base-chinese", use_safetensors=True).to(device)

# state_dict = torch.load("check_point/finetune_detect_1_0.ckpt")
# detection.load_state_dict(state_dict)

train_dataset = CSCDataset("dataset/original_sighan_train.tsv")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)

learning_rate = 1e-5
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(detection.parameters(), lr=learning_rate)

total = len(train_dataloader)
epochs = 2
for epoch in range(epochs):
    with tqdm(total=total) as progress:
        for i, (original_text, correct_text, error_label, pinyin_ids) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = tokenizer(original_text, return_tensors='pt', max_length=256, truncation=True, padding=True).to(device)
            pinyin_ids = pinyin_ids[:, :inputs["input_ids"].size(-1)].to(device)
            error_prob = error_label[:, :inputs["input_ids"].size(-1)].to(device)
    
            # output = corrector(**inputs, pinyin_ids=pinyin_ids, error_prob=error_prob)[0]
            output = detection(**inputs).logits
            
            loss = loss_fn(output.permute(0, 2, 1), error_prob)
            loss.backward()
            optimizer.step()
    
            # if i%10 == 0:
            progress.set_postfix(loss=loss.item(), refresh=False)
            progress.update(1)

    torch.save(detection.state_dict(), f"no_pretrain/detect_model_epoch_{epoch}")

