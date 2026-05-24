import os
import json
import re
import httpx
from typing import List, Dict, Tuple, Optional

class AINerInterface:
    def __init__(self, model_type: str = "deepseek"):
        self.model_type = model_type.lower()
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/v1"
        self.ollama_url = "http://localhost:11434/api/generate"

    def refine_ner(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not base_entities:
            return []

        if self.model_type == "deepseek":
            return self._refine_with_deepseek(text, base_entities)
        elif self.model_type == "openai" or self.model_type == "gpt":
            return self._refine_with_openai(text, base_entities)
        elif self.model_type == "ollama":
            return self._refine_with_ollama(text, base_entities)
        else:
            return base_entities

    def _build_prompt(self, text: str, entities: List[Tuple[str, str]]) -> str:
        entity_str = "\n".join([f"- {ent_type}: {entity}" for entity, ent_type in entities])
        return f"""你是一个专业的中文命名实体识别助手。给定以下文本和初步识别出的实体，请根据上下文语境判断并修正实体类型。

文本: {text}

初步识别的实体:
{entity_str}

请根据语境分析每个实体的正确类型。可能的实体类型包括：
- 人名（人名）：人物的真实姓名
- 公司（公司）：公司、企业名称
- 组织（组织）：正式组织、机构
- 地址（地址）：地点、地址
- 书名（书名）：书籍名称
- 游戏（游戏）：游戏名称
- 政府（政府）：政府机构
- 电影（电影）：电影名称
- 职位（职位）：官职、职称
- 景点（景点）：旅游景点
- 动物（动物）：动物名称
- 植物（植物）：植物名称
- 水果（水果）：水果名称
- 食物（食物）：食物、食品
- 品牌（品牌）：产品品牌名称
- 产品（产品）：具体产品名称

注意事项：
1. 如果某个实体可能是多种类型，请根据上下文语境选择最合适的一种。
2. 如果两个实体存在包含关系（如"苹果公司"包含"苹果"），只保留更完整的实体。
3. 如果实体明显错误（如单个字"机"被识别为地址），请删除或修正它。
4. "苹果手机"是产品，不是公司或水果。
5. "苹果公司"是公司，"苹果"在商业语境下可能是品牌。
6. "人民大会堂"是地址/景点，不要拆分为"人民大会"。

请以JSON格式输出修正后的实体列表，格式如下：
{{
  "refined_entities": [
    {{"entity": "实体文本", "type": "实体类型(中文)"}},
    ...
  ]
}}

只输出JSON，不要有其他内容。"""

    async def _call_deepseek_api(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_openai_api(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_ollama_api(self, prompt: str) -> str:
        data = {
            "model": "qwen2.5:latest",
            "prompt": prompt,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.ollama_url, json=data)
            response.raise_for_status()
            return response.json()["response"]

    def _parse_json_response(self, response: str) -> List[Tuple[str, str]]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = []
                for item in data.get("refined_entities", []):
                    entities.append((item["entity"], item["type"]))
                return entities
        except json.JSONDecodeError:
            pass
        return []

    async def refine_ner_async(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not base_entities or not self.api_key:
            return base_entities

        prompt = self._build_prompt(text, base_entities)

        try:
            if self.model_type == "deepseek":
                response = await self._call_deepseek_api(prompt)
            elif self.model_type == "openai" or self.model_type == "gpt":
                response = await self._call_openai_api(prompt)
            elif self.model_type == "ollama":
                response = await self._call_ollama_api(prompt)
            else:
                return base_entities

            return self._parse_json_response(response)
        except Exception as e:
            print(f"[WARNING] AI refinement failed: {e}")
            return base_entities

    def refine_ner_sync(self, text: str, base_entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.refine_ner_async(text, base_entities)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.refine_ner_async(text, base_entities))
        except Exception as e:
            print(f"[WARNING] Sync AI refinement failed: {e}")
            return base_entities

class AINerConfig:
    TYPE_MAPPING = {
        "人名": "name",
        "公司": "company",
        "组织": "organization",
        "地址": "address",
        "书名": "book",
        "游戏": "game",
        "政府": "government",
        "电影": "movie",
        "职位": "position",
        "景点": "scene",
        "动物": "animal",
        "植物": "plant",
        "水果": "fruit",
        "食物": "food",
        "品牌": "brand",
        "产品": "product"
    }

    @classmethod
    def to_english_type(cls, cn_type: str) -> str:
        return cls.TYPE_MAPPING.get(cn_type, cn_type)

    @classmethod
    def to_chinese_type(cls, en_type: str) -> str:
        for cn, en in cls.TYPE_MAPPING.items():
            if en == en_type:
                return cn
        return en_type
