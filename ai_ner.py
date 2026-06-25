import os
import json
import re
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Set


class AINerInterface:
    """增强版AI命名实体识别接口 - 专注于语义推理和上下文判断"""
    
    # 预定义的实体类型（不扩展）
    VALID_ENTITY_TYPES = {
        "name": "姓名",
        "company": "公司",
        "brand": "品牌",
        "product": "产品",
        "address": "地址",
        "organization": "组织",
        "government": "政府",
        "position": "职位",
        "scene": "景点",
        "book": "书名",
        "movie": "电影",
        "game": "游戏",
        "animal": "动物",
        "plant": "植物",
        "food": "食物",
        "event": "事件",
        "time": "时间",
        "date": "日期",
        "number": "数字"
    }
    
    def __init__(self, model_type: str = "ollama"):
        self.model_type = model_type.lower()
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/v1"
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # 如果没有设置API Key，自动切换到Ollama
        if not self.api_key and self.model_type == "deepseek":
            print("[INFO] 未设置DeepSeek API Key，自动切换到Ollama本地模型")
            self.model_type = "ollama"
        
        # 预定义的语义推理规则库
        self.semantic_rules = self._build_semantic_rules()
    
    def _build_semantic_rules(self) -> Dict[str, List[Dict]]:
        """构建语义推理规则库"""
        return {
            "company": [
                {
                    "contexts": ["公司", "集团", "企业", "有限公司", "股份有限公司", "科技", "互联网", "软件"],
                    "examples": ["阿里巴巴", "腾讯", "华为", "百度", "京东", "字节跳动", "美团", "滴滴"]
                },
                {
                    "contexts": ["CEO", "董事长", "总裁", "总经理", "创始人", "CEO", "CTO", "COO"],
                    "examples": ["阿里巴巴", "腾讯", "华为", "小米", "百度"]
                }
            ],
            "brand": [
                {
                    "contexts": ["品牌", "产品", "手机", "电脑", "电脑", "汽车", "家电"],
                    "examples": ["华为", "小米", "苹果", "三星", "OPPO", "vivo", "联想", "戴尔", "惠普"]
                }
            ],
            "product": [
                {
                    "contexts": ["产品", "发布", "推出", "新款", "旗舰", "系统"],
                    "examples": ["鸿蒙", "安卓", "iOS", "Windows", "Linux", "麒麟", "骁龙"]
                }
            ],
            "name": [
                {
                    "contexts": ["说", "表示", "认为", "指出", "强调", "介绍", "提到", "谈及", "采访", "对话", "访问", "会见", "会见"],
                    "examples": []  # 人名通过上下文判断
                },
                {
                    "contexts": ["先生", "女士", "教授", "博士", "老师", "总", "总理", "主席", "部长"],
                    "examples": []
                }
            ],
            "movie": [
                {
                    "contexts": ["电影", "影片", "导演", "主演", "上映", "票房", "片名", "观影"],
                    "examples": ["流浪地球", "满江红", "你好，李焕英", "哪吒", "战狼", "长津湖", "复仇者联盟", "阿凡达", "泰坦尼克号"]
                }
            ],
            "book": [
                {
                    "contexts": ["书籍", "图书", "作者", "出版", "名著", "小说", "作品"],
                    "examples": ["红楼梦", "西游记", "三国演义", "水浒传", "活着", "平凡的世界"]
                }
            ],
            "game": [
                {
                    "contexts": ["游戏", "玩家", "电竞", "网游", "手游", "端游", "发布", "steam"],
                    "examples": ["王者荣耀", "英雄联盟", "原神", "明日之后", "和平精英", "我的世界", "塞尔达", "马里奥"]
                }
            ],
            "address": [
                {
                    "contexts": ["位于", "地址", "座落于", "在", "市", "省", "县", "区", "街道", "路"],
                    "examples": []
                }
            ],
            "organization": [
                {
                    "contexts": ["组织", "机构", "协会", "学会", "联盟", "委员会"],
                    "examples": []
                }
            ],
            "government": [
                {
                    "contexts": ["政府", "部门", "机关", "局", "厅", "部", "委", "办"],
                    "examples": []
                }
            ],
            "scene": [
                {
                    "contexts": ["景区", "景点", "旅游", "公园", "博物馆", "展览", "著名", "游览"],
                    "examples": ["故宫", "长城", "兵马俑", "黄山", "西湖", "张家界", "九寨沟"]
                }
            ],
            "position": [
                {
                    "contexts": ["担任", "任职", "职位", "职务", "工作"],
                    "examples": ["CEO", "董事长", "总裁", "总经理", "教授", "工程师", "总监"]
                }
            ],
            "food": [
                {
                    "contexts": ["美食", "菜肴", "餐厅", "菜", "特色", "小吃", "传统"],
                    "examples": ["火锅", "烤鸭", "川菜", "粤菜", "鲁菜", "淮扬菜", "湘菜"]
                }
            ],
            "animal": [
                {
                    "contexts": ["动物", "宠物", "野生动物", "哺乳动物", "鸟类", "鱼类"],
                    "examples": ["狗", "猫", "熊猫", "老虎", "狮子", "大象", "猴子", "兔子"]
                }
            ],
            "plant": [
                {
                    "contexts": ["植物", "花卉", "树木", "花草", "绿化", "园林"],
                    "examples": ["玫瑰", "牡丹", "菊花", "梅花", "樱花", "银杏", "松树"]
                }
            ],
            "event": [
                {
                    "contexts": ["会议", "活动", "赛事", "展览", "论坛", "峰会", "运动会"],
                    "examples": ["奥运会", "世界杯", "亚运会", "世博会", "两会", "互联网大会"]
                }
            ]
        }
    
    def refine_ner(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """使用AI进行实体识别增强（本地语义推理）"""
        if not base_entities:
            return self._rule_based_recognition(text)
        
        return self._semantic_inference_with_context(text, base_entities)
    
    async def refine_ner_async(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """异步版本的实体识别增强"""
        return self.refine_ner(text, base_entities)
    
    def refine_ner_with_pos(self, text: str, base_entities: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
        """使用位置信息进行实体识别增强"""
        if not base_entities:
            return []
        
        result = []
        seen_positions = set()
        
        for entity_type, entity, pos in base_entities:
            entity_positions = set(range(pos, pos + len(entity)))
            overlap = False
            for p in entity_positions:
                if p in seen_positions:
                    overlap = True
                    break
            
            if overlap:
                continue
            
            refined_type = self._smart_type_refinement(entity_type, entity, text, pos)
            
            if refined_type:
                result.append((refined_type, entity, pos))
            else:
                result.append((entity_type, entity, pos))
            
            for p in entity_positions:
                seen_positions.add(p)
        
        return result
    
    def _rule_based_recognition(self, text: str) -> List[Tuple[str, str]]:
        """基于规则的实体识别"""
        entities = []
        
        # 应用每个类型的语义规则
        for entity_type, rules in self.semantic_rules.items():
            for rule in rules:
                for context in rule["contexts"]:
                    if context in text:
                        # 找到上下文位置，提取附近可能的实体
                        for example in rule["examples"]:
                            if example in text and (entity_type, example) not in entities:
                                entities.append((entity_type, example))
                        break
        
        return entities
    
    def _semantic_inference(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """语义推理增强"""
        enhanced_entities = []
        existing_entities = set(base_entities)
        
        for entity_type, entity in base_entities:
            if (entity_type, entity) not in enhanced_entities:
                enhanced_entities.append((entity_type, entity))
            
            if entity_type in self.semantic_rules:
                for rule in self.semantic_rules[entity_type]:
                    for context in rule["contexts"]:
                        if context in text:
                            for example in rule["examples"]:
                                if example in text and (entity_type, example) not in existing_entities:
                                    if (entity_type, example) not in enhanced_entities:
                                        enhanced_entities.append((entity_type, example))
                            break
        
        return enhanced_entities
    
    def _semantic_inference_with_context(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """基于上下文的语义推理增强"""
        result = []
        seen_positions = set()
        
        for entity_type, entity in base_entities:
            start = 0
            while True:
                idx = text.find(entity, start)
                if idx == -1:
                    break
                
                entity_positions = set(range(idx, idx + len(entity)))
                overlap = False
                for p in entity_positions:
                    if p in seen_positions:
                        overlap = True
                        break
                
                if overlap:
                    start = idx + 1
                    continue
                
                refined_type = self._smart_type_refinement(entity_type, entity, text, idx)
                
                if refined_type:
                    result.append((refined_type, entity))
                else:
                    result.append((entity_type, entity))
                
                for p in entity_positions:
                    seen_positions.add(p)
                
                start = idx + 1
        
        return result
    
    def _smart_type_refinement(self, original_type: str, entity: str, text: str, position: int) -> str:
        """智能类型修正 - 仅在有明确证据时才修改类型"""
        window_size = 10
        start_idx = max(0, position - window_size)
        end_idx = min(len(text), position + len(entity) + window_size)
        context = text[start_idx:end_idx]
        
        if entity == "苹果":
            if "吃" in context or "剩" in context or "食物" in context or "水果" in context or "削皮" in context or "咬" in context:
                return "food"
            if "公司" in context or "研发" in context or "总部" in context or "CEO" in context:
                return "company"
            if entity + "手机" in text and position + len(entity) < len(text) and text[position + len(entity)] == "手":
                return "product"
            if "手机" in context or "电脑" in context or "iPad" in context or "Mac" in context:
                return "brand"
        
        if entity == "华为":
            if "升华" in context:
                return ""
        
        if original_type == "name":
            return original_type
        
        if original_type == "address":
            return original_type
        
        return original_type
    
    def _call_ollama(self, prompt: str) -> str:
        """调用Ollama本地模型"""
        try:
            import httpx
            import json
            
            payload = {
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False
            }
            
            response = httpx.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"[ERROR] Ollama调用失败: {response.status_code}")
                return ""
        except Exception as e:
            print(f"[ERROR] Ollama调用异常: {e}")
            return ""
    
    def _call_deepseek(self, prompt: str) -> str:
        """调用DeepSeek API"""
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"[ERROR] DeepSeek API调用失败: {response.status_code}")
                return ""
        except Exception as e:
            print(f"[ERROR] DeepSeek API调用异常: {e}")
            return ""
    
    def get_entity_type_cn(self, entity_type: str) -> str:
        """获取实体类型的中文名称"""
        return self.VALID_ENTITY_TYPES.get(entity_type, entity_type)
