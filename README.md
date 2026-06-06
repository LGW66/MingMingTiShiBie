# 中文命名实体识别系统 (Chinese NER System)

基于BERT模型的中文命名实体识别系统，支持AI增强识别和联网搜索功能。

## 功能特性

- ✅ **BERT模型**：基于BERT-base-chinese的高性能命名实体识别
- ✅ **AI增强**：支持Ollama本地模型、DeepSeek API、OpenAI API进行实体类型修正
- ✅ **联网搜索**：集成DuckDuckGo搜索，提高实体识别准确性
- ✅ **长篇文档处理**：支持句子分割、逐句识别、结果整合
- ✅ **多格式支持**：支持txt、md、docx文档上传识别
- ✅ **前后端分离**：FastAPI后端 + 现代化前端界面
- ✅ **19种实体类型**：姓名、公司、品牌、产品、地址、组织、政府、职位、景点、书名、电影、游戏、动物、植物、食物、事件、时间、日期、数字

## 技术栈

- **后端**：FastAPI + PyTorch + BERT
- **前端**：HTML5 + CSS3 + JavaScript
- **AI集成**：Ollama / DeepSeek API / OpenAI API
- **联网搜索**：DuckDuckGo Search

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

服务将在 `http://127.0.0.1:8081` 启动。

### 访问界面

打开浏览器访问 `http://127.0.0.1:8081` 即可使用。

## API接口

### POST /api/ner

命名实体识别接口

**请求体**:
```json
{
    "text": "待识别的文本",
    "use_ai": false,
    "is_document": false,
    "use_search": false
}
```

**参数说明**:
- `text`: 待识别的文本内容
- `use_ai`: 是否使用AI增强（需配置Ollama或API密钥）
- `is_document`: 是否为长篇文档模式
- `use_search`: 是否启用联网搜索（需配合use_ai使用）

**响应**:
```json
{
    "entities": [
        {
            "type": "name",
            "type_cn": "姓名",
            "value": "李白"
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
- `ai_model`: AI模型类型（ollama/deepseek/openai）

## 配置说明

### AI模型配置

#### 使用Ollama本地模型（推荐）

1. 安装Ollama：https://ollama.com/
2. 拉取模型：`ollama pull qwen2.5`
3. 启动Ollama服务：`ollama serve`

#### 使用DeepSeek API

设置环境变量：
```bash
export DEEPSEEK_API_KEY=your_api_key
```

#### 使用OpenAI API

设置环境变量：
```bash
export OPENAI_API_KEY=your_api_key
```

## 项目结构

```
MingMingTiShiBie-main/
├── app.py              # FastAPI主应用
├── ai_ner.py           # AI增强模块
├── config.py           # 配置文件
├── model.py            # BERT模型定义
├── data_utils.py       # 数据处理工具
├── document_processor.py # 文档处理模块
├── requirements.txt    # 依赖列表
├── static/             # 静态资源
│   └── index.html      # 前端界面
├── models/             # 模型文件目录
└── data/               # 数据集目录
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

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
