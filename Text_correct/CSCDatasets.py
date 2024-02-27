import torch
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, file, max_len=256):
        super().__init__()
        with open(file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        self.max_len = max_len
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        text, label = self.lines[i].strip('\n').split('\t')
        label = eval(label)
        if len(label) > self.max_len - 2:
            text = text[:self.max_len - 2]
            label = label[:self.max_len - 2]
        pad_len = self.max_len - len(text) - 2
        label = torch.tensor([0] + label + [0] * (pad_len + 1))
        return text, label

