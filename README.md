# 中文命名实体识别系统 (Chinese NER System)

基于 BiLSTM+CRF 模型的中文命名实体识别系统，支持18种实体类型识别，并集成AI增强功能。

## 功能特性

- 🏷️ **支持18种实体类型**：姓名、公司、品牌、产品、地址、组织、政府、职位、景点、书名、电影、游戏、动物、植物、食物、事件、时间、日期、数字
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
| 姓名 | 李白、小明、毛泽东 |
| 公司 | 苹果公司、阿里巴巴、腾讯 |
| 品牌 | 苹果、华为、小米 |
| 产品 | 苹果手机、iPhone、MacBook |
| 地址 | 北京、上海、人民大会堂 |
| 组织 | 联合国、北约、世卫组织 |
| 政府 | 中华人民共和国、国务院 |
| 职位 | 主席、教授、总理 |
| 景点 | 故宫、长城、埃菲尔铁塔 |
| 书名 | 红楼梦、围城、三国演义 |
| 电影 | 泰坦尼克号、阿凡达 |
| 游戏 | 王者荣耀、英雄联盟 |
| 动物 | 狗、猫、老虎、熊猫 |
| 植物 | 花、草、树、玫瑰 |
| 食物 | 苹果、香蕉、面包、饺子 |
| 事件 | 人民代表大会、奥运会、亚运会 |
| 时间 | 早上、下午、8点、14:00 |
| 日期 | 今天、明天、2024年1月1日 |
| 数字 | 100元、五个、百分之五十 |

## 配置说明

### 环境变量

- `DEEPSEEK_API_KEY`: DeepSeek API Key
- `OPENAI_API_KEY`: OpenAI API Key

### Ollama 配置

```bash
# 安装 Ollama (Windows)
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
├── config.py                # 配置文件（实体类型、知识库等）
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
>>> 李白在家吃苹果
   姓名：李白
   食物：苹果

>>> 苹果公司发布了新款手机
   公司：苹果公司
   产品：手机

>>> 毛泽东主席在第一届人民代表大会发表讲话
   姓名：毛泽东
   职位：主席
   事件：第一届人民代表大会

>>> 明天早上8点开会
   日期：明天
   时间：早上
   时间：8点
```

## 上传到GitHub步骤

1. **初始化Git仓库**
```bash
cd d:\TRAxiangmu\gcsj
git init
git lfs install
git lfs track "models/*.pt"
git lfs track "models/*.pkl"
```

2. **添加文件**
```bash
git add .
```

3. **提交**
```bash
git commit -m "Add Chinese NER System with 18 entity types and AI enhancement"
```

4. **连接远程仓库**
```bash
git remote add origin https://github.com/LGW66/MingMingTiShiBie.git
```

5. **推送到GitHub**
```bash
git push -u origin main
```

## 许可证

MIT License
