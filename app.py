from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os
import torch
import asyncio

from config import *
from model import BertNER
from ai_ner import AINerInterface
from document_processor import split_into_sentences

app = FastAPI(title="中文命名实体识别系统", description="基于BERT的中文命名实体识别API")

# 启动时检查
print("="*60)
print("*** 服务启动检查 ***")
print(f"KNOWLEDGE_BASE 类别数量: {len(KNOWLEDGE_BASE)}")
print(f"华为是否在company中: {'华为' in KNOWLEDGE_BASE.get('company', [])}")
print(f"EN_TO_CN 长度: {len(EN_TO_CN)}")
print("="*60)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        from transformers import AutoTokenizer
        import requests
        
        # 先尝试从本地加载tokenizer和模型
        local_model_path = os.path.join(os.path.dirname(__file__), 'bert_cache')
        if os.path.exists(local_model_path):
            print(f"[INFO] 从本地加载tokenizer和模型: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            # 尝试从huggingface加载
            try:
                requests.head("https://huggingface.co", timeout=3)
                print("[INFO] 网络连接正常，尝试加载tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            except Exception as net_err:
                print(f"[WARNING] 网络连接失败: {net_err}")
                print("[INFO] 跳过tokenizer和模型加载，仅使用知识库")
                model = None
                tokenizer = None
                return
        
        best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
        if not os.path.exists(best_model_path):
            print("[WARNING] 未找到模型文件，请先训练模型")
            return
        
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        
        # 使用本地模型路径或默认模型名称
        model_path = local_model_path if os.path.exists(local_model_path) else BERT_MODEL_NAME
        model = BertNER(
            num_labels=NUM_LABELS,
            model_name=model_path
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"[INFO] 模型加载成功 (epoch {checkpoint['epoch'] + 1})")
    except Exception as e:
        print(f"[WARNING] 模型加载失败: {e}")
        print("[INFO] 将仅使用知识库进行实体识别")
        model = None
        tokenizer = None

def predict_single_sentence(text: str) -> List[Tuple[str, str]]:
    if not model:
        return []
    
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    predictions = model.predict(input_ids, attention_mask)
    predictions = predictions[0].cpu().numpy().tolist()
    
    input_ids_np = encoding['input_ids'][0].cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_np)
    
    start_idx = tokens.index('[CLS]') + 1 if '[CLS]' in tokens else 0
    end_idx = tokens.index('[SEP]') if '[SEP]' in tokens else len(tokens)
    
    entities = []
    current_entity = []
    current_type = None
    
    for i in range(start_idx, end_idx):
        token = tokens[i]
        pred = predictions[i]
        label = ID_TO_LABEL[pred]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
            current_entity = [token]
            current_type = label[2:]
        elif label.startswith('I-') and current_entity and label[2:] == current_type:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((current_type, ''.join(current_entity)))
                current_entity = []
                current_type = None
    
    if current_entity:
        entities.append((current_type, ''.join(current_entity)))
    
    entities = [(typ, entity.replace('##', '')) for typ, entity in entities]
    
    name_suffixes_to_remove = ['在', '是', '有', '和', '与', '及', '了', '的', '着', '过']
    processed_entities = []
    for typ, entity in entities:
        if typ == 'name':
            while len(entity) > 1 and entity[-1] in name_suffixes_to_remove:
                entity = entity[:-1]
        if len(entity) >= 1:
            processed_entities.append((typ, entity))
    
    return processed_entities

def knowledge_based_ner(text: str) -> List[Tuple[str, str]]:
    entities = []
    
    for entity_type, entity_list in KNOWLEDGE_BASE.items():
        for entity in entity_list:
            if len(entity) >= 2 and entity in text:
                entities.append((entity_type, entity))
            elif len(entity) == 1 and entity in text:
                if entity in ['狗', '猫', '花', '草', '树', '鱼', '鸟', '虫']:
                    entities.append((entity_type, entity))
    
    entities.extend(extract_dates(text))
    
    return entities

def extract_dates(text: str) -> List[Tuple[str, str]]:
    import re
    dates = []
    
    date_patterns = [
        r'(\d{4}年\d{1,2}月\d{1,2}日)',
        r'(\d{4}年\d{1,2}月)',
        r'(\d{4}/\d{1,2}/\d{1,2})',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{1,2}月\d{1,2}日)',
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            dates.append(('date', match))
    
    return dates

def remove_invalid_single_chars(entities, text):
    filtered = []
    invalid_chars = ['的', '在', '是', '了', '和', '与', '或', '都', '也', '很', '把', '被', '让', '给', '用', '对', '这', '那', '你', '我', '他', '她', '它', '们', '有', '没', '要', '能', '会', '可', '以', '说', '看', '想', '知', '道', '来', '去', '进', '出', '上', '下', '左', '右', '前', '后', '里', '外', '大', '小', '多', '少', '好', '坏', '新', '旧', '高', '低', '长', '短', '快', '慢', '轻', '重', '冷', '热', '南', '北', '东', '西', '机', '话', '号', '本', '为', '于', '从', '当', '其', '此', '每', '各', '所']
    for entity_type, entity in entities:
        if len(entity) == 1 and entity in invalid_chars:
            continue
        filtered.append((entity_type, entity))
    return filtered

def context_based_filter(entities, text):
    filtered = []
    blacklist_contexts = {
        '华为': ['升华'],
    }
    
    for entity_type, entity in entities:
        should_filter = False
        
        if entity in blacklist_contexts:
            for context in blacklist_contexts[entity]:
                if context in text:
                    index = text.find(entity)
                    context_index = text.find(context)
                    if context_index >= 0 and abs(index - context_index) <= 2:
                        should_filter = True
                        break
        
        if not should_filter:
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

class NERRequest(BaseModel):
    text: str
    use_ai: Optional[bool] = False
    is_document: Optional[bool] = False

class Entity(BaseModel):
    type: str
    type_cn: str
    value: str

class NERResponse(BaseModel):
    entities: List[Entity]
    used_ai: bool
    total_count: int
    processed_sentences: Optional[int] = None

@app.post("/api/ner", response_model=NERResponse)
async def recognize_entities(request: NERRequest):
    try:
        text = request.text
        processed_sentences = None
        
        # 非常显眼的调试日志
        print("="*50)
        print("*** NER API 请求已收到 ***")
        print(f"请求文本 (repr): {repr(text)}")
        print(f"is_document: {request.is_document}")
        print(f"use_ai: {request.use_ai}")
        print("="*50)
        
        # 测试知识库是否正常
        test_entities = []
        for entity_type, entity_list in KNOWLEDGE_BASE.items():
            for entity in entity_list:
                if entity in text:
                    test_entities.append((entity_type, entity))
                    print(f"[DEBUG] 知识库找到: {entity} ({entity_type})")
        print(f"[DEBUG] 测试知识库找到总数: {len(test_entities)}")
        
        if request.is_document:
            sentences = split_into_sentences(text)
            all_entities = []
            
            for sentence in sentences:
                model_entities = predict_single_sentence(sentence)
                knowledge_entities = knowledge_based_ner(sentence)
                
                merged_entities = merge_entities(model_entities, knowledge_entities, sentence)
                merged_entities = add_all_possible_types(merged_entities, sentence)
                merged_entities = remove_invalid_single_chars(merged_entities, sentence)
                merged_entities = context_based_filter(merged_entities, sentence)
                
                all_entities.extend(merged_entities)
            
            final_entities = deduplicate_by_context(all_entities, text)
            processed_sentences = len(sentences)
        else:
            model_entities = predict_single_sentence(text)
            knowledge_entities = knowledge_based_ner(text)
            
            # 调试日志
            print(f"[DEBUG] 文本: {text}")
            print(f"[DEBUG] 模型识别: {len(model_entities)} 个")
            print(f"[DEBUG] 知识库识别: {len(knowledge_entities)} 个 - {knowledge_entities}")
            
            merged_entities = merge_entities(model_entities, knowledge_entities, text)
            print(f"[DEBUG] 合并后: {len(merged_entities)} 个 - {merged_entities}")
            
            merged_entities = add_all_possible_types(merged_entities, text)
            print(f"[DEBUG] 添加类型后: {len(merged_entities)} 个")
            
            merged_entities = remove_invalid_single_chars(merged_entities, text)
            print(f"[DEBUG] 移除单字后: {len(merged_entities)} 个")
            
            merged_entities = context_based_filter(merged_entities, text)
            print(f"[DEBUG] 上下文过滤后: {len(merged_entities)} 个")
            
            final_entities = deduplicate_by_context(merged_entities, text)
            print(f"[DEBUG] 去重后: {len(final_entities)} 个 - {final_entities}")
        
        if request.use_ai:
            ai_interface = AINerInterface()
            final_entities = await ai_interface.refine_ner_async(text, final_entities)
        
        formatted_entities = []
        for entity_type, entity_value in final_entities:
            formatted_entities.append(Entity(
                type=entity_type,
                type_cn=EN_TO_CN.get(entity_type, entity_type),
                value=entity_value
            ))
        
        response = NERResponse(
            entities=formatted_entities,
            used_ai=request.use_ai,
            total_count=len(formatted_entities),
            processed_sentences=processed_sentences
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
async def test_endpoint():
    print("[DEBUG] 测试端点被调用")
    return {"message": "测试成功", "status": "OK"}

@app.post("/api/ner/file")
async def recognize_file(file: UploadFile = File(...), use_ai: Optional[bool] = False, ai_model: Optional[str] = "ollama"):
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        sentences = split_into_sentences(text)
        all_entities = []
        
        for sentence in sentences:
            model_entities = predict_single_sentence(sentence)
            knowledge_entities = knowledge_based_ner(sentence)
            
            merged_entities = merge_entities(model_entities, knowledge_entities, sentence)
            merged_entities = add_all_possible_types(merged_entities, sentence)
            merged_entities = remove_invalid_single_chars(merged_entities, sentence)
            
            all_entities.extend(merged_entities)
        
        final_entities = deduplicate_by_context(all_entities, text)
        
        if use_ai:
            ai_interface = AINerInterface(ai_model)
            final_entities = await ai_interface.refine_ner_async(text, final_entities)
        
        formatted_entities = []
        for entity_type, entity_value in final_entities:
            formatted_entities.append(Entity(
                type=entity_type,
                type_cn=EN_TO_CN.get(entity_type, entity_type),
                value=entity_value
            ))
        
        return NERResponse(
            entities=formatted_entities,
            used_ai=use_ai,
            total_count=len(formatted_entities),
            processed_sentences=len(sentences)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    load_model()
    print("[INFO] 启动FastAPI服务...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, reload=False)
