import os
import torch
import asyncio

from config import *
from data_utils import get_dataloaders
from model import BiLSTM_CRF
from ai_ner import AINerInterface

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                if entity in ['狗', '猫', '花', '草', '树', '鱼', '鸟', '虫', '米', '面', '肉', '茶', '酒', '糖', '盐', '油']:
                    entities.append((entity_type, entity))

    return entities

def get_chinese_entity_type(en_type):
    return EN_TO_CN.get(en_type, en_type)

def remove_invalid_single_chars(entities, text):
    filtered = []
    for entity_type, entity in entities:
        if len(entity) == 1 and entity in ['的', '在', '是', '了', '和', '与', '或', '都', '也', '很', '把', '被', '让', '给', '用', '对', '这', '那', '你', '我', '他', '她', '它', '们', '有', '没', '要', '能', '会', '可', '以', '说', '看', '想', '知', '道', '来', '去', '进', '出', '上', '下', '左', '右', '前', '后', '里', '外', '大', '小', '多', '少', '好', '坏', '新', '旧', '高', '低', '长', '短', '快', '慢', '轻', '重', '冷', '热', '南', '北', '东', '西', '机', '话', '号', '本', '为', '于', '从', '当', '其', '此', '每', '各', '所']:
            continue
        filtered.append((entity_type, entity))
    return filtered

def merge_entities(model_entities, knowledge_entities, text):
    merged = []
    seen = set()

    all_entities = model_entities + knowledge_entities
    all_entities = sorted(all_entities, key=lambda x: text.index(x[1]) if x[1] in text else -1)

    for entity_type, entity in all_entities:
        if entity not in seen:
            merged.append((entity_type, entity))
            seen.add(entity)

    return merged

def add_all_possible_types(entities, text):
    entity_dict = {}
    for entity_type, entity in entities:
        if entity not in entity_dict:
            entity_dict[entity] = set()
        entity_dict[entity].add(entity_type)

    for entity in list(entity_dict.keys()):
        for kb_type, kb_list in KNOWLEDGE_BASE.items():
            if entity in kb_list and kb_type not in entity_dict[entity]:
                entity_dict[entity].add(kb_type)

    result = []
    for entity, types in entity_dict.items():
        for entity_type in types:
            result.append((entity_type, entity))

    return sorted(result, key=lambda x: text.index(x[1]) if x[1] in text else -1)

def deduplicate_by_context(entities, text):
    entity_list = sorted(entities, key=lambda x: (-len(x[1]), text.index(x[1]) if x[1] in text else -1))

    filtered = []
    for entity_type, entity in entity_list:
        is_substring = False
        for _, existing_entity in filtered:
            if entity in existing_entity and entity != existing_entity:
                is_substring = True
                break
        if not is_substring:
            filtered.append((entity_type, entity))

    entity_dict = {}
    for entity_type, entity in filtered:
        if entity not in entity_dict:
            entity_dict[entity] = []
        if entity_type not in entity_dict[entity]:
            entity_dict[entity].append(entity_type)

    result = []
    for entity, types in entity_dict.items():
        for t in types:
            result.append((t, entity))

    result = sorted(result, key=lambda x: text.index(x[1]) if x[1] in text else -1)
    return result

async def refine_with_ai_async(text, entities, ai_interface):
    if not entities:
        return entities

    try:
        refined = await ai_interface.refine_ner_async(text, entities)
        return refined
    except Exception as e:
        print(f"[WARNING] AI refinement failed: {e}")
        return entities

def main():
    print("=" * 60)
    print("    中文命名实体识别系统 v3.0 (BiLSTM+CRF + AI增强)")
    print("=" * 60)
    print("支持的实体类型：")
    print("  姓名、公司、组织、地址、书名、游戏、政府、电影、职位、景点")
    print("  动物、植物、食物、品牌、产品、事件、时间、日期、数字")
    print("=" * 60)
    print("\nAI模型选项：")
    print("  0 - 不使用AI增强（直接使用模型识别）")
    print("  1 - DeepSeek API")
    print("  2 - OpenAI GPT API")
    print("  3 - Ollama本地模型")
    print("=" * 60)

    ai_choice = input("\n请选择AI模型 (0-3): ").strip()
    ai_interface = None
    use_ai = False

    if ai_choice == "1":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            api_key = input("请输入DeepSeek API Key: ").strip()
            os.environ["DEEPSEEK_API_KEY"] = api_key
        ai_interface = AINerInterface("deepseek")
        use_ai = True
        print("[INFO] 使用 DeepSeek AI 增强识别")
    elif ai_choice == "2":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = input("请输入OpenAI API Key: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key
        ai_interface = AINerInterface("openai")
        use_ai = True
        print("[INFO] 使用 OpenAI GPT AI 增强识别")
    elif ai_choice == "3":
        ai_interface = AINerInterface("ollama")
        use_ai = True
        print("[INFO] 使用 Ollama 本地模型增强识别")
    else:
        print("[INFO] 不使用AI增强，直接使用模型识别")

    print("\n" + "=" * 60)

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

            merged_entities = merge_entities(model_entities, knowledge_entities, text)

            merged_entities = add_all_possible_types(merged_entities, text)

            merged_entities = remove_invalid_single_chars(merged_entities, text)

            merged_entities = deduplicate_by_context(merged_entities, text)

            if use_ai and ai_interface:
                print("\n[INFO] 正在使用AI增强识别...")
                refined_entities = asyncio.run(refine_with_ai_async(text, merged_entities, ai_interface))
                if refined_entities:
                    merged_entities = refined_entities

            print("\n识别结果：")
            if merged_entities:
                for entity_type, entity in merged_entities:
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
