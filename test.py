import os
import torch

from config import *
from data_utils import get_dataloaders
from model import BertNER
from train import compute_metrics


def test(model, test_loader, device, id_to_label):
    model.eval()
    all_predictions = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_masks.extend(attention_mask.cpu().numpy().tolist())
    
    precision, recall, f1, micro_f1 = compute_metrics(all_predictions, all_labels, all_masks, id_to_label)
    
    print("Test Results:")
    print(f"Micro F1: {micro_f1:.4f}")
    print("\nPer-label metrics:")
    for label in sorted(id_to_label.values()):
        print(f"{label}: P={precision[label]:.4f}, R={recall[label]:.4f}, F1={f1[label]:.4f}")
    
    return micro_f1


def predict_single_sentence(model, text, tokenizer, label_map, id_to_label, device):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    predictions = model.predict(input_ids, attention_mask, token_type_ids)
    
    entities = []
    current_entity = []
    current_type = None
    
    for token, pred in zip(tokens, predictions[0][1:-1]):
        label = id_to_label[pred]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
            current_entity = [token]
            current_type = label[2:]
        elif label.startswith('I-') and current_entity:
            current_entity.append(token)
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
    train_loader, dev_loader, test_loader, tokenizer = get_dataloaders(BATCH_SIZE, MAX_LEN)
    
    model = BertNER(
        num_labels=NUM_LABELS,
        model_name=BERT_MODEL_NAME
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
        entities = predict_single_sentence(model, sentence, tokenizer, LABEL_MAP, ID_TO_LABEL, DEVICE)
        print(f"\nSentence: {sentence}")
        print("Entities:")
        for entity_type, entity in entities:
            cn_type = get_chinese_entity_type(entity_type)
            print(f"  {cn_type}：{entity}")


if __name__ == '__main__':
    main()