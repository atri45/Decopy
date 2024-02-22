from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        with open(file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i].strip('\n')
        inputs, labels = line.split('\t')
        return inputs, labels
