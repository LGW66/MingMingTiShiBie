import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item.get('label', {})
        
        tokens = []
        token_labels = []
        
        for char in text:
            sub_tokens = self.tokenizer.tokenize(char)
            tokens.extend(sub_tokens)
            if len(sub_tokens) > 0:
                token_labels.extend([0] * len(sub_tokens))
        
        for entity_type, entities in labels.items():
            for entity_info in entities.values():
                for start, end in entity_info:
                    entity_text = text[start:end+1]
                    entity_tokens = self.tokenizer.tokenize(entity_text)
                    token_start = 0
                    found = False
                    
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            token_start = i
                            found = True
                            break
                    
                    if found:
                        prefix = 'B-' if entity_type.startswith('B-') else 'B-'
                        actual_type = entity_type.split('-')[-1] if '-' in entity_type else entity_type
                        token_labels[token_start] = self.label_map.get(f'{prefix}{actual_type}', 0)
                        for j in range(1, len(entity_tokens)):
                            token_labels[token_start + j] = self.label_map.get(f'I-{actual_type}', 0)
        
        tokens = tokens[:self.max_len-2]
        token_labels = token_labels[:self.max_len-2]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        labels = [0] + token_labels + [0]
        
        padding_len = self.max_len - len(input_ids)
        input_ids += [0] * padding_len
        attention_mask += [0] * padding_len
        token_type_ids += [0] * padding_len
        labels += [0] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def load_json(file_name):
    file_path = os.path.join('./data', file_name)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def get_dataloaders(batch_size=16, max_len=128):
    from config import TRAIN_FILE, DEV_FILE, TEST_FILE, LABEL_MAP, BERT_MODEL_NAME
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_data = load_json(TRAIN_FILE)
    dev_data = load_json(DEV_FILE)
    test_data = load_json(TEST_FILE)
    
    train_dataset = NERDataset(train_data, tokenizer, LABEL_MAP, max_len)
    dev_dataset = NERDataset(dev_data, tokenizer, LABEL_MAP, max_len)
    test_dataset = NERDataset(test_data, tokenizer, LABEL_MAP, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader, tokenizer