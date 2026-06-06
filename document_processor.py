import os
import re
try:
    import docx
except ImportError:
    docx = None

from config import MAX_LEN, MIN_SENTENCE_LEN, MAX_SENTENCE_LEN, PARAGRAPH_MERGE_LEN


def load_document(file_path):
    """支持多种文档格式：txt, md, docx"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    if file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif file_ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif file_ext == '.docx' and docx:
        doc = docx.Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def split_into_sentences(text, max_len=None):
    """将文档分割成句子，优化超长文本处理
    
    Args:
        text: 输入文本
        max_len: 最大句子长度（字符数），默认从config读取
    
    Returns:
        句子列表
    """
    if max_len is None:
        max_len = MAX_SENTENCE_LEN
    
    # 清理文本：规范化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 第一步：按段落分割（保留段落结构）
    paragraphs = re.split(r'\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    sentences = []
    for para in paragraphs:
        # 短段落合并，避免产生太短的句子
        if len(para) < PARAGRAPH_MERGE_LEN and paragraphs.index(para) < len(paragraphs) - 1:
            # 与下一个段落合并
            next_para = paragraphs[paragraphs.index(para) + 1]
            para = para + ' ' + next_para
            paragraphs[paragraphs.index(para) + 1] = para  # 更新下一个段落，避免重复
        
        # 第二步：按句子分隔符分割
        temp_sentences = re.split(r'[。！？；]+', para)
        temp_sentences = [s.strip() for s in temp_sentences if s.strip()]
        
        for sentence in temp_sentences:
            # 第三步：处理超长句子
            if len(sentence) <= max_len:
                sentences.append(sentence)
            else:
                # 智能分割：优先在标点处分割
                sub_sentences = smart_split_long_sentence(sentence, max_len)
                sentences.extend(sub_sentences)
    
    # 过滤空句子和太短的句子
    sentences = [s for s in sentences if len(s) >= MIN_SENTENCE_LEN]
    
    return sentences


def smart_split_long_sentence(text, max_len):
    """智能分割超长句子
    
    Args:
        text: 超长句子
        max_len: 最大长度
    
    Returns:
        分割后的句子列表
    """
    result = []
    
    # 优先分割位置：逗号、顿号、连接词
    split_patterns = [
        r'，',      # 逗号
        r'、',      # 顿号
        r'而且',    # 转折
        r'但是',    # 转折
        r'然而',    # 转折
        r'所以',    # 因果
        r'因此',    # 因果
        r'于是',    # 因果
        r'然后',    # 顺序
        r'接着',    # 顺序
        r'并且',    # 并列
        r'同时',    # 并列
        r'此外',    # 补充
        r'另外',    # 补充
        r'特别是',  # 强调
        r'尤其是',  # 强调
    ]
    
    while len(text) > max_len:
        best_split_pos = -1
        best_split_char = ''
        
        # 查找最优分割点（优先在max_len附近找分割符）
        search_start = max(0, max_len - 50)
        search_end = min(len(text), max_len + 50)
        
        for pattern in split_patterns:
            matches = list(re.finditer(pattern, text[search_start:search_end]))
            if matches:
                # 选择最接近max_len的分割点
                for match in matches:
                    pos = search_start + match.start()
                    if best_split_pos == -1 or abs(pos - max_len) < abs(best_split_pos - max_len):
                        best_split_pos = pos
                        best_split_char = match.group()
        
        if best_split_pos == -1:
            # 没有找到合适的分割点，强制在max_len处分割
            result.append(text[:max_len])
            text = text[max_len:]
        else:
            # 在分割点后分割（保留分割符）
            result.append(text[:best_split_pos + len(best_split_char)])
            text = text[best_split_pos + len(best_split_char):]
    
    # 添加剩余文本
    if text:
        result.append(text)
    
    return result


def split_into_sentences_with_info(text, max_len=None):
    """将文档分割成句子，并返回详细信息
    
    Returns:
        dict: {
            'sentences': [...],
            'total_chars': int,
            'total_sentences': int,
            'avg_sentence_len': float
        }
    """
    sentences = split_into_sentences(text, max_len)
    
    return {
        'sentences': sentences,
        'total_chars': len(text),
        'total_sentences': len(sentences),
        'avg_sentence_len': len(text) / len(sentences) if sentences else 0
    }


def batch_process(files):
    """批量处理多个文档"""
    results = []
    for file_path in files:
        try:
            text = load_document(file_path)
            sentences = split_into_sentences(text)
            results.append({
                'file': file_path,
                'text': text,
                'sentences': sentences,
                'info': split_into_sentences_with_info(text)
            })
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            results.append({
                'file': file_path,
                'error': str(e)
            })
    return results


def export_results(results, output_file):
    """将识别结果导出为JSON文件"""
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已导出到 {output_file}")
