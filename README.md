<<<<<<< HEAD
# 中文命名实体识别系统 (Chinese NER System)

基于 BiLSTM+CRF 模型的中文命名实体识别系统，支持16种实体类型识别，并集成AI增强功能。

## 功能特性

- 🏷️ **支持16种实体类型**：姓名、公司、品牌、产品、地址、组织、政府、职位、景点、书名、电影、游戏、动物、植物、水果、食物
- 🤖 **AI增强识别**：支持 DeepSeek、OpenAI GPT、Ollama 本地模型
- 📊 **交互式界面**：提供友好的命令行交互体验
- 🚀 **Web服务部署**：支持 FastAPI 部署（可扩展）

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- FastAPI（可选）
- Gradio（可选）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

### 2. 交互式测试

```bash
python interactive.py          # 基础版本
python interactive_ai.py       # AI增强版本
```

### 3. 选择AI模型

运行 `interactive_ai.py` 后，选择：
- `0` - 不使用AI增强
- `1` - DeepSeek API
- `2` - OpenAI GPT API
- `3` - Ollama 本地模型

## 支持的实体类型

| 类型 | 示例 |
|------|------|
| 姓名 | 李白、小明 |
| 公司 | 苹果公司、阿里巴巴 |
| 品牌 | 苹果、华为 |
| 产品 | 苹果手机、iPhone |
| 地址 | 北京、人民大会堂 |
| 组织 | 联合国、北约 |
| 政府 | 中华人民共和国 |
| 职位 | 主席、教授 |
| 景点 | 故宫、长城 |
| 书名 | 红楼梦、围城 |
| 电影 | 泰坦尼克号、阿凡达 |
| 游戏 | 王者荣耀、英雄联盟 |
| 动物 | 狗、猫、老虎 |
| 植物 | 花、草、树 |
| 水果 | 苹果、香蕉 |
| 食物 | 面包、饺子 |

## 配置说明

### 环境变量

- `DEEPSEEK_API_KEY`: DeepSeek API Key
- `OPENAI_API_KEY`: OpenAI API Key

### Ollama 配置

```bash
# 安装 Ollama
iwr https://ollama.ai/install.sh | sh

# 下载模型
ollama pull qwen2.5:latest
```

## 项目结构

```
.
├── data/                    # 训练数据
│   ├── train_new.json       # 训练集
│   ├── dev_new.json         # 验证集
│   └── test.json            # 测试集
├── models/                  # 模型文件
│   ├── best_model.pt        # 最佳模型
│   ├── last_checkpoint.pt   # 最后检查点
│   └── vocab.pkl            # 词汇表
├── ai_ner.py                # AI增强模块
├── config.py                # 配置文件
├── data_utils.py            # 数据处理工具
├── interactive.py           # 基础交互界面
├── interactive_ai.py        # AI增强交互界面
├── model.py                 # BiLSTM+CRF模型
├── requirements.txt         # 依赖列表
├── test.py                  # 测试脚本
└── train.py                 # 训练脚本
```

## 使用示例

```
>>> 李白在家玩苹果公司研发的苹果手机
   姓名：李白
   公司：苹果公司
   产品：苹果手机

>>> 习近平主席在人民大会堂发表讲话
   姓名：习近平
   职位：主席
   地址：人民大会堂
```

## 上传到GitHub步骤

1. **初始化Git仓库**
```bash
cd d:\TRAxiangmu\gcsj
git init
```

2. **创建 .gitignore 文件**
```bash
echo "__pycache__/" > .gitignore
echo "models/*.pt" >> .gitignore
echo "models/*.pkl" >> .gitignore
echo ".env" >> .gitignore
```

3. **添加文件**
```bash
git add .
```

4. **提交**
```bash
git commit -m "Initial commit: Chinese NER System with BiLSTM+CRF and AI enhancement"
```

5. **连接远程仓库**
```bash
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

6. **推送到GitHub**
```bash
git branch -M main
git push -u origin main
```

## 许可证

MIT License
=======
# MingMingTiShiBie
工程实践4
>>>>>>> a5e93edd45360a693b92ab72b6fc5cb0a6fecac8
