"""测试AI增强NER功能"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ai_ner import AINerInterface

def test_ai_ner():
    print("=" * 60)
    print("测试AI增强NER功能")
    print("=" * 60)
    
    # 创建AI接口
    ai_interface = AINerInterface(model_type="ollama", enable_search=False)
    print(f"使用模型: {ai_interface.model_type}")
    
    # 测试文本
    test_text = "《明日之后》是一部非常好看的电影"
    
    # 基础实体（模拟基础模型识别结果）
    base_entities = [
        ("movie", "明日之后"),
    ]
    
    print(f"\n测试文本: {test_text}")
    print(f"基础实体: {base_entities}")
    print("\n开始AI增强...")
    
    # 调用AI增强
    try:
        result = ai_interface.refine_ner(test_text, base_entities, use_search=False)
        print(f"\n成功! 结果: {result}")
    except Exception as e:
        print(f"\n失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_ner()
