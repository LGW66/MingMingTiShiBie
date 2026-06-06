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


def context_based_type_correction(entities, text):
    CONTEXT_KEYWORDS = {
        'fruit': ['吃', '吃的', '吃了', '吃下', '咬', '品尝', '水果', '果汁', '水果摊', '水果店', '水果篮', '新鲜', '甜', '酸', '味道', '口感'],
        'food': ['吃', '喝', '烹饪', '做饭', '餐厅', '饭', '菜', '汤', '肉', '蛋', '面', '米', '厨房', '餐具', '美味', '食谱'],
        'animal': ['宠物', '猫', '狗', '鸟', '鱼', '动物园', '兽医', '毛茸茸', '可爱', '尾巴', '爪子', '叫声', '跑', '跳', '飞', '游'],
        'plant': ['花', '草', '树', '叶子', '根', '茎', '花园', '植物', '盆栽', '种植', '浇水', '施肥', '发芽', '开花'],
        'company': ['公司', '企业', '科技', '产品', '手机', '电脑', '软件', '互联网', '上市', '股票', 'CEO', '创始人', '总部'],
        'brand': ['品牌', '牌子', '标志', '商标', '设计', '广告', '代言', '限量', '经典', '新款'],
        'product': ['手机', '电脑', '平板', '手表', '耳机', '电视', '冰箱', '空调', '汽车', '型号'],
    }
    
    entity_dict = {}
    for entity_type, entity in entities:
        if entity not in entity_dict:
            entity_dict[entity] = []
        if entity_type not in entity_dict[entity]:
            entity_dict[entity].append(entity_type)
    
    corrected_entities = []
    
    for entity, types in entity_dict.items():
        if len(types) == 1:
            corrected_entities.append((types[0], entity))
        else:
            best_type = types[0]
            max_score = 0
            
            for entity_type in types:
                score = 0
                if entity_type in CONTEXT_KEYWORDS:
                    for keyword in CONTEXT_KEYWORDS[entity_type]:
                        if keyword in text:
                            score += 1
                if entity_type == 'fruit' and entity in ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '梨', '桃']:
                    if any(action in text for action in ['吃', '吃的', '吃了', '吃下', '咬', '品尝']):
                        score += 2
                if entity_type == 'company' and entity in ['苹果', '谷歌', '微软', '华为', '阿里巴巴', '腾讯']:
                    if any(tech in text for tech in ['公司', '科技', '手机', '电脑', '软件']):
                        score += 2
                if score > max_score:
                    max_score = score
                    best_type = entity_type
            
            corrected_entities.append((best_type, entity))
    
    return sorted(corrected_entities, key=lambda x: text.index(x[1]) if x[1] in text else -1)


def main():
    print("=" * 60)
    print("    中文命名实体识别系统 v2.0 (BiLSTM+CRF + 知识库增强)")
    print("=" * 60)
    print("支持的实体类型：")
    print("  姓名、公司、组织、地址、书名、游戏、政府、电影、职位、景点")
    print("  动物、植物、食物、品牌、产品、事件、时间、日期、数字")
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

            corrected_entities = context_based_type_correction(unique_entities, text)

            print("\n识别结果：")
            if corrected_entities:
                for entity_type, entity in corrected_entities:
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
