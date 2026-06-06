# 中文命名实体识别系统 (Chinese NER System)

基于BERT模型的中文命名实体识别系统，支持AI语义推理增强。

## 功能特性

- ✅ **BERT模型**：基于BERT-base-chinese的高性能命名实体识别
- ✅ **AI增强**：支持Ollama本地模型进行语义推理和上下文判断
- ✅ **知识库增强**：包含5000+实体，覆盖19种实体类型
- ✅ **长篇文档处理**：支持句子分割、逐句识别、结果整合
- ✅ **多格式支持**：支持txt、md、docx文档上传识别
- ✅ **前后端分离**：FastAPI后端 + 现代化前端界面
- ✅ **19种实体类型**：姓名、公司，品牌、产品、地址、组织，政府、职位、景点、书名、电影、游戏、动物、植物、食物、事件、时间、日期、数字

## 技术栈

- **后端**：FastAPI + PyTorch + BERT
- **前端**：HTML5 + CSS3 + JavaScript
- **AI集成**：Ollama (本地模型)
- **模型**：bert-base-chinese

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/LGW66/MingMingTiShiBie.git
cd MingMingTiShiBie
```

### 2. 下载模型和数据

由于GitHub对文件大小有限制，你需要手动下载以下文件：

#### BERT模型和Tokenizer
从HuggingFace下载bert-base-chinese模型到 `bert_cache/` 目录：
```bash
mkdir -p bert_cache
# 使用Python下载
python -c "from transformers import BertModel, AutoTokenizer; \
    m = BertModel.from_pretrained('bert-base-chinese'); \
    t = AutoTokenizer.from_pretrained('bert-base-chinese'); \
    m.save_pretrained('./bert_cache'); \
    t.save_pretrained('./bert_cache')"
```

#### 训练好的模型（可选）
将训练好的模型文件放到 `models/` 目录：
- `models/best_model.pt` - 最佳模型检查点
- `models/last_checkpoint.pt` - 最后检查点
- `models/vocab.pkl` - 词汇表

#### 训练数据（可选）
将训练数据放到 `data/` 目录：
- `data/train_new_extended.json` - 训练集
- `data/dev_new.json` - 验证集
- `data/test.json` - 测试集

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
python app.py
```

服务将在 `http://127.0.0.1:8081` 启动。

### 5. 访问界面

打开浏览器访问 `http://127.0.0.1:8081` 即可使用。

## API接口

### POST /api/ner

命名实体识别接口

**请求体**:
```json
{
    "text": "待识别的文本",
    "use_ai": false,
    "is_document": false
}
```

**参数说明**:
- `text`: 待识别的文本内容
- `use_ai`: 是否使用AI增强
- `is_document`: 是否为长篇文档模式

**响应**:
```json
{
    "entities": [
        {
            "type": "name",
            "type_cn": "姓名",
            "value": "马云"
        }
    ],
    "used_ai": false,
    "total_count": 1,
    "processed_sentences": 1
}
```

### POST /api/ner/file

文件上传识别接口

**请求**:
- `file`: 文档文件（支持txt、md、docx）
- `use_ai`: 是否使用AI增强
- `ai_model`: AI模型类型（ollama）

## AI增强功能配置

### 使用Ollama本地模型（推荐）

1. 安装Ollama：https://ollama.com/
2. 拉取模型：`ollama pull qwen2.5`
3. 启动Ollama服务：`ollama serve`

AI增强功能会使用本地模型进行语义推理，提高实体识别的准确性。

## 项目结构

```
MingMingTiShiBie/
├── app.py              # FastAPI主应用
├── ai_ner.py           # AI增强模块（语义推理）
├── config.py           # 配置文件
├── model.py            # BERT模型定义
├── data_utils.py       # 数据处理工具
├── document_processor.py # 文档处理模块
├── requirements.txt    # 依赖列表
├── static/             # 静态资源
│   └── index.html      # 前端界面
├── bert_cache/         # BERT模型和Tokenizer（需手动下载）
├── models/             # 训练好的模型（可选）
└── data/              # 数据集（可选）
```

## 实体类型

| 英文 | 中文 | 说明 |
|------|------|------|
| name | 姓名 | 人物的真实姓名 |
| company | 公司 | 公司、企业名称 |
| brand | 品牌 | 产品品牌名称 |
| product | 产品 | 具体产品名称 |
| address | 地址 | 地点、地址 |
| organization | 组织 | 正式组织、机构 |
| government | 政府 | 政府机构 |
| position | 职位 | 官职、职称 |
| scene | 景点 | 旅游景点 |
| book | 书名 | 书籍名称 |
| movie | 电影 | 电影名称 |
| game | 游戏 | 游戏名称 |
| animal | 动物 | 动物名称 |
| plant | 植物 | 植物名称 |
| food | 食物 | 食物、食品 |
| event | 事件 | 会议、活动、赛事 |
| time | 时间 | 时刻、时段 |
| date | 日期 | 具体日期 |
| number | 数字 | 数量、金额、序号 |

## 训练自己的模型

如果你想训练自己的模型：

1. 准备训练数据（JSON格式）
2. 修改 `config.py` 中的配置
3. 运行训练脚本：
```bash
python train.py
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
