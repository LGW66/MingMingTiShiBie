import os
import torch
import pickle

from config import *
from data_utils import get_dataloaders
from model import BiLSTM_CRF
from train import compute_metrics


def test(model, test_loader, device, id_to_label):
    model.eval()
    all_predictions = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for char_ids, label_ids, masks in test_loader:
            char_ids = char_ids.to(device)
            label_ids = label_ids.to(device)
            masks = masks.to(device)
            
            predictions = model.predict(char_ids, masks)
            
            all_predictions.extend(predictions)
            all_labels.extend(label_ids.cpu().numpy().tolist())
            all_masks.extend(masks.cpu().numpy().tolist())
    
    precision, recall, f1, micro_f1 = compute_metrics(all_predictions, all_labels, all_masks, id_to_label)
    
    print("Test Results:")
    print(f"Micro F1: {micro_f1:.4f}")
    print("\nPer-label metrics:")
    for label in sorted(id_to_label.values()):
        print(f"{label}: P={precision[label]:.4f}, R={recall[label]:.4f}, F1={f1[label]:.4f}")
    
    return micro_f1


def predict_single_sentence(model, text, char_to_idx, id_to_label, device):
    char_ids = [char_to_idx.get(c, char_to_idx['<UNK>']) for c in text]
    char_ids = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.ones(len(text), dtype=torch.bool).unsqueeze(0).to(device)
    
    predictions = model.predict(char_ids, mask)
    
    entities = []
    current_entity = []
    current_type = None
    
    for char, pred in zip(text, predictions[0]):
        label = id_to_label[pred]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
            current_entity = [char]
            current_type = label[2:]
        elif label.startswith('I-') and current_entity:
            current_entity.append(char)
        else:
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
                current_entity = []
                current_type = None
    
    if current_entity:
        entities.append((current_type, ''.join(current_entity)))
    
    return entities


def get_chinese_entity_type(en_type):
    return EN_TO_CN.get(en_type, en_type)


def main():
    _, _, test_loader, vocab, char_to_idx = get_dataloaders(BATCH_SIZE)
    
    vocab_size = len(vocab)
    
    model = BiLSTM_CRF(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=NUM_LABELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
    else:
        print("No best model found. Please train the model first.")
        return
    
    test(model, test_loader, DEVICE, ID_TO_LABEL)
    
    print("\nTesting single sentence:")
    test_sentences = [
        "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
        "布鲁京斯研究所桑顿中国中心研究部主任李成说，东亚的和平与安全，是美国的核心利益之一。",
        "我在北京工作，公司是阿里巴巴。"
    ]
    
    for sentence in test_sentences:
        entities = predict_single_sentence(model, sentence, char_to_idx, ID_TO_LABEL, DEVICE)
        print(f"\nSentence: {sentence}")
        print("Entities:")
        for entity_type, entity in entities:
            cn_type = get_chinese_entity_type(entity_type)
            print(f"  {cn_type}：{entity}")


if __name__ == '__main__':
    main()
