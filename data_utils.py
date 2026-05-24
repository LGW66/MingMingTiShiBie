import json
import os
import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, LABEL_MAP, ID_TO_LABEL


def load_json(file_name):
    data = []
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_to_bio(data):
    bio_data = []
    for item in data:
        text = item['text']
        labels = item.get('label', {})
        
        bio_labels = ['O'] * len(text)
        
        for entity_type, entities in labels.items():
            for entity_name, spans in entities.items():
                for span in spans:
                    start, end = span
                    for i in range(start, end + 1):
                        if i == start:
                            bio_labels[i] = f'B-{entity_type}'
                        else:
                            bio_labels[i] = f'I-{entity_type}'
        
        bio_data.append((list(text), bio_labels))
    return bio_data


def build_vocab(bio_data, min_freq=1):
    char_counter = Counter()
    for chars, _ in bio_data:
        char_counter.update(chars)
    
    vocab = ['<PAD>', '<UNK>']
    for char, freq in char_counter.items():
        if freq >= min_freq:
            vocab.append(char)
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    return vocab, char_to_idx


class NERDataset(Dataset):
    def __init__(self, bio_data, char_to_idx, label_map):
        self.bio_data = bio_data
        self.char_to_idx = char_to_idx
        self.label_map = label_map
    
    def __len__(self):
        return len(self.bio_data)
    
    def __getitem__(self, idx):
        chars, labels = self.bio_data[idx]
        
        char_ids = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in chars]
        label_ids = [self.label_map.get(l, self.label_map['O']) for l in labels]
        
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)


def collate_fn(batch):
    char_ids_list, label_ids_list = zip(*batch)
    
    max_len = max(len(ids) for ids in char_ids_list)
    
    char_ids_padded = []
    label_ids_padded = []
    masks = []
    
    for char_ids, label_ids in zip(char_ids_list, label_ids_list):
        padding_len = max_len - len(char_ids)
        char_ids_padded.append(torch.cat([char_ids, torch.zeros(padding_len, dtype=torch.long)]))
        label_ids_padded.append(torch.cat([label_ids, torch.zeros(padding_len, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(len(char_ids), dtype=torch.bool), torch.zeros(padding_len, dtype=torch.bool)]))
    
    return torch.stack(char_ids_padded), torch.stack(label_ids_padded), torch.stack(masks)


def get_dataloaders(batch_size=32):
    train_data = load_json('train.json')
    dev_data = load_json('dev.json')
    test_data = load_json('test.json')
    
    train_bio = convert_to_bio(train_data)
    dev_bio = convert_to_bio(dev_data)
    test_bio = convert_to_bio(test_data)
    
    vocab, char_to_idx = build_vocab(train_bio)
    
    os.makedirs('./models', exist_ok=True)
    with open('./models/vocab.pkl', 'wb') as f:
        pickle.dump((vocab, char_to_idx), f)
    
    train_dataset = NERDataset(train_bio, char_to_idx, LABEL_MAP)
    dev_dataset = NERDataset(dev_bio, char_to_idx, LABEL_MAP)
    test_dataset = NERDataset(test_bio, char_to_idx, LABEL_MAP)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, vocab, char_to_idx


if __name__ == '__main__':
    train_loader, dev_loader, test_loader, vocab, char_to_idx = get_dataloaders()
    print(f"Vocab size: {len(vocab)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Dev samples: {len(dev_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    for batch in train_loader:
        char_ids, label_ids, masks = batch
        print(f"char_ids shape: {char_ids.shape}")
        print(f"label_ids shape: {label_ids.shape}")
        print(f"masks shape: {masks.shape}")
        break
