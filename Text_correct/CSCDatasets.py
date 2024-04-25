import torch
from torch.utils.data import Dataset

class CSCDataset(Dataset):
    def __init__(self, file, max_len=256):
        super().__init__()
        with open(file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        original_text, correct_text, error_label, pinyin_ids = self.lines[i].strip('\n').split('\t')
        error_label = eval(error_label)
        pinyin_ids = eval(pinyin_ids)
        pad_len = self.max_len - len(pinyin_ids) - 2
        error_label = torch.tensor([0] + error_label + [0] +[-100] * pad_len)
        pinyin_ids = torch.tensor([0] + pinyin_ids + [0] * (pad_len + 1))
        return original_text, correct_text, error_label, pinyin_ids
