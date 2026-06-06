"""测试Ollama连接"""
import os
import httpx
import asyncio

async def test_ollama():
    ollama_url = "http://localhost:11434/api/generate"
    
    # 获取模型名称
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    print(f"测试模型: {model}")
    
    # 测试简单的生成请求
    data = {
        "model": model,
        "prompt": "你好",
        "stream": False
    }
    
    try:
        print("正在连接Ollama...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ollama_url, json=data)
            response.raise_for_status()
            result = response.json()
            print(f"成功! 响应: {result.get('response', '无响应')[:100]}")
    except Exception as e:
        print(f"失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
