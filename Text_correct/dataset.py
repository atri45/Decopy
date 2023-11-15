from torch.utils.data.dataset import Dataset

class PinyinMaskedLMDataset(Dataset):
    def __init__(self, file_path:str):
        super().__init__()
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i].strip('\n')
        inputs, labels = line.split('\t')
        return inputs, labels
