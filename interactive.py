import os
import torch
import re

from config import *
from data_utils import get_dataloaders
from model import BiLSTM_CRF


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


def knowledge_based_ner(text):
    entities = []
    text_lower = text.lower()
    
    for entity_type, entity_list in KNOWLEDGE_BASE.items():
        for entity in entity_list:
            if len(entity) >= 2 and entity in text:
                entities.append((entity_type, entity))
            elif len(entity) == 1 and entity in text:
                if entity in ['狗', '猫', '花', '草', '树', '鱼', '鸟', '虫']:
                    entities.append((entity_type, entity))
    
    return entities


def get_chinese_entity_type(en_type):
    return EN_TO_CN.get(en_type, en_type)


def main():
    print("=" * 60)
    print("    中文命名实体识别系统 v2.0 (BiLSTM+CRF + 知识库增强)")
    print("=" * 60)
    print("支持的实体类型：")
    print("  姓名、公司、组织、地址、书名、游戏、政府、电影、职位、景点")
    print("=" * 60)
    
    _, _, _, vocab, char_to_idx = get_dataloaders(BATCH_SIZE)
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
        model.eval()
        print(f"[INFO] 模型加载成功 (epoch {checkpoint['epoch'] + 1})")
    else:
        print("[ERROR] 未找到模型文件，请先训练模型")
        return
    
    print("\n请输入要识别的文本（输入 'exit' 或 'quit' 退出）：")
    
    while True:
        try:
            text = input("\n>>> ")
            
            if text.strip().lower() in ['exit', 'quit', '退出']:
                print("感谢使用，再见！")
                break
            
            if not text.strip():
                print("请输入有效文本")
                continue
            
            model_entities = predict_single_sentence(model, text, char_to_idx, ID_TO_LABEL, DEVICE)
            
            knowledge_entities = knowledge_based_ner(text)
            
            model_entities_set = set(model_entities)
            for entity_type, entity in knowledge_entities:
                if (entity_type, entity) not in model_entities_set:
                    model_entities.append((entity_type, entity))
            
            model_entities = sorted(model_entities, key=lambda x: text.index(x[1]))

            entity_dict = {}
            for entity_type, entity in model_entities:
                if entity not in entity_dict:
                    entity_dict[entity] = []
                if entity_type not in entity_dict[entity]:
                    entity_dict[entity].append(entity_type)

            unique_entities = []
            seen = set()
            for entity, types in entity_dict.items():
                is_substring = False
                for existing_entity in seen:
                    if entity in existing_entity or (len(entity) == 1 and existing_entity.startswith(entity)):
                        is_substring = True
                        break
                if not is_substring:
                    seen.add(entity)
                    for t in types:
                        unique_entities.append((t, entity))

            print("\n识别结果：")
            if unique_entities:
                for entity_type, entity in unique_entities:
                    cn_type = get_chinese_entity_type(entity_type)
                    print(f"  {cn_type}：{entity}")
            else:
                print("  未识别到实体")
                
        except KeyboardInterrupt:
            print("\n感谢使用，再见！")
            break
        except Exception as e:
            print(f"发生错误：{e}")


if __name__ == '__main__':
    main()
