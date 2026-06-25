from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os
import torch

from config import *
from data_utils import get_dataloaders
from model import BiLSTM_CRF

app = FastAPI(title="中文命名实体识别系统", description="基于BiLSTM+CRF的中文命名实体识别API")

model = None
char_to_idx = None

def load_model():
    global model, char_to_idx
    _, _, _, vocab, char_to_idx = get_dataloaders(BATCH_SIZE)
    
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    if not os.path.exists(best_model_path):
        print("[WARNING] 未找到模型文件，请先训练模型")
        return
    
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    
    checkpoint_vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
    checkpoint_num_labels = checkpoint['model_state_dict']['fc.weight'].shape[0]
    
    print(f"[INFO] Checkpoint信息 - 词汇表大小: {checkpoint_vocab_size}, 标签数量: {checkpoint_num_labels}")
    print(f"[INFO] 当前配置 - 词汇表大小: {len(vocab)}, 标签数量: {NUM_LABELS}")
    
    if checkpoint_vocab_size != len(vocab) or checkpoint_num_labels != NUM_LABELS:
        print(f"[WARNING] 配置不匹配，将使用checkpoint中的结构参数")
    
    model = BiLSTM_CRF(
        vocab_size=checkpoint_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=checkpoint_num_labels,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] 模型加载成功 (epoch {checkpoint['epoch'] + 1})")

def predict_single_sentence(text: str) -> List[Tuple[str, str]]:
    if not model:
        load_model()
    
    char_ids = [char_to_idx.get(c, char_to_idx['<UNK>']) for c in text]
    char_ids = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mask = torch.ones(len(text), dtype=torch.bool).unsqueeze(0).to(DEVICE)
    
    predictions = model.predict(char_ids, mask)
    
    entities = []
    current_entity = []
    current_type = None
    
    for char, pred in zip(text, predictions[0]):
        label = ID_TO_LABEL[pred]
        
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

def knowledge_based_ner(text: str) -> List[Tuple[str, str]]:
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

class NerRequest(BaseModel):
    text: str

class Entity(BaseModel):
    type: str
    type_cn: str
    value: str

class NerResponse(BaseModel):
    text: str
    entities: List[Entity]
    highlighted_text: str

@app.post("/api/ner", response_model=NerResponse)
async def ner(request: NerRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="请输入有效文本")
        
        model_entities = predict_single_sentence(text)
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
        
        entities = []
        for entity_type, entity in unique_entities:
            entities.append({
                "type": entity_type,
                "type_cn": EN_TO_CN.get(entity_type, entity_type),
                "value": entity
            })
        
        highlighted_text = text
        entity_colors = {
            "name": "#FFE4E1",
            "company": "#E0FFE0",
            "brand": "#E0E0FF",
            "product": "#FFFFE0",
            "address": "#FFE4C4",
            "organization": "#E0FFFF",
            "government": "#FFD700",
            "position": "#FFB6C1",
            "scene": "#98FB98",
            "book": "#DDA0DD",
            "movie": "#87CEEB",
            "game": "#F0E68C",
            "animal": "#90EE90",
            "plant": "#228B22",
            "fruit": "#FFA500",
            "food": "#CD5C5C"
        }
        
        sorted_entities = sorted(unique_entities, key=lambda x: -len(x[1]))
        for entity_type, entity in sorted_entities:
            color = entity_colors.get(entity_type, "#FFFF00")
            highlighted_text = highlighted_text.replace(
                entity,
                f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{entity}</mark>'
            )
        
        return {
            "text": text,
            "entities": entities,
            "highlighted_text": highlighted_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/entity_types")
async def get_entity_types():
    return {
        "entity_types": [{"en": k, "cn": v} for k, v in EN_TO_CN.items()]
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    load_model()
    print("[INFO] 启动FastAPI服务...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)