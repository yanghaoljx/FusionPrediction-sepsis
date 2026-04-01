import json
import random
import torch
from torch.utils.data import Dataset
import os
random.seed(42)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_dataset(data, label_map, seed=42):
    # 检查标签是否合法
    for example in data:
        assert 0 <= example['label'] < len(label_map), f"Label out of range: {example['label']}"
    
    random.seed(seed)
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)
    
    train_dataset = data[:train_end]
    val_dataset = data[train_end:val_end]
    test_dataset = data[val_end:]
    
    return train_dataset, val_dataset, test_dataset

def get_datasets(data_path, label_map, seed=42):
    data = read_jsonl(data_path)
    return split_dataset(data, label_map, seed)


class MyDataset(Dataset):
    def __init__(self, mode, base_path):
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val' or 'test'"
        data_path = os.path.join(base_path, mode)
        rep_file = os.path.join(data_path, f'sepsis_{mode}_reps.pt')
        label_file = os.path.join(data_path, f'sepsis_{mode}_labels.pt')

        self.sents_reps = torch.load(rep_file)
        self.sents_reps = torch.mean(self.sents_reps, dim=1)  # (N, 4096)
        self.labels = torch.load(label_file)
        self.sample_num = self.labels.shape[0]
        
    def __getitem__(self, index):
        return self.sents_reps[index],self.labels[index]        
        
    def __len__(self):
        return self.sample_num