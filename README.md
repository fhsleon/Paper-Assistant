# 论文智能助手

基于 LangChain 的论文智能问答系统，支持论文上传、RAG 检索问答、论文搜索、图表分析等功能。

## 功能特性

- 📄 **论文管理**：上传 PDF 论文，自动解析和索引
- 🔍 **智能问答**：基于 RAG 的论文内容问答
- 🌐 **论文搜索**：集成 OpenAlex API 搜索学术论文
- 🖼️ **图表分析**：多模态模型分析论文中的图表
- 💬 **Web 界面**：Streamlit 构建的聊天式交互界面

## 技术栈

- **后端**：FastAPI + LangChain 1.2.6
- **前端**：Streamlit
- **模型**：本地模型Qwen2.5-3B-Instruct（文本）+ Qwen2.5-VL-3B-Instruct（视觉）
- **向量数据库**：FAISS + BM25 混合检索
- **Embedding**：BGE-M3

## 项目结构

```
paper_rag_chat/
├── api.py                 # FastAPI 后端
├── web.py                 # Streamlit 前端
├── core/
│   ├── agent.py           # ReAct Agent 核心
│   ├── rag.py             # RAG 检索模块
│   └── intent_classifier.py
├── tools/
│   ├── paper_tools.py     # 论文搜索工具
│   └── vision_tools.py    # 图表分析工具
├── config/
│   ├── prompts.py         # 提示词配置
│   └── utils.py           # 工具函数
└── parsers/
    └── pdf_parser.py      # PDF 解析器
```

---

## 开发过程中遇到的问题与解决方案

### 一、模型推理问题

#### 1. 模型输出循环重复

**问题描述**：
模型生成回复时陷入循环，重复相似内容。

**解决方案**：
调整生成参数：
```python
pipe = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=512,       # 限制输出长度
    temperature=0.1,          # 降低随机性
    top_p=0.9,
    repetition_penalty=1.1,   # 添加重复惩罚
    do_sample=True
)
```

---

#### 2. HuggingFacePipeline 返回类型不一致

**问题描述**：
`AttributeError: 'str' object has no attribute 'content'`

**原因**：
不同 LLM 类型的 `invoke` 返回值不同：
- `ChatOllama` 返回 `AIMessage` 对象，有 `.content` 属性
- `HuggingFacePipeline` 直接返回字符串

**解决方案**：
```python
response = self.llm.invoke(prompt)
if hasattr(response, 'content'):
    result = response.content
else:
    result = str(response)
```

---

### 二、RAG 与向量数据库问题

#### 3. 向量数据库重复加载

**问题描述**：
每次加载同一篇论文都要重新构建向量索引，耗时较长。

**解决方案**：
实现基于文件哈希的缓存机制：
```python
def _get_cache_key(self, file_path: str) -> str:
    stat = os.stat(file_path)
    return hashlib.md5(f"{file_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()

def _load_cache(self, cache_key: str) -> bool:
    if os.path.exists(cache_path):
        self.vectorstore = FAISS.load_local(cache_path, self.embeddings)
        return True
    return False
```

---

### 三、意图识别问题

#### 4. 意图分类错误

**问题描述**：
- 用户问"这篇论文的创新点是什么"被分类为 SEARCH_PAPERS
- 用户问"帮我搜一下 U-Net 改进"被分类为 ASK_PAPER

**原因**：
提示词不够明确，模型无法区分"针对已有论文提问"和"搜索新论文"。

**解决方案**：
重写提示词，添加明确的规则和示例：
```python
INTENT_PROMPT = """
判断用户意图：
- ASK_PAPER：针对已上传论文提问（关键词："这篇"、"它"、"论文的xxx"）
- SEARCH_PAPERS：搜索新论文（关键词："帮我找"、"有没有"、"搜索"）

示例：
"这篇论文的创新点" -> ASK_PAPER
"帮我搜 U-Net 改进" -> SEARCH_PAPERS
"""
```

---

#### 5. OpenAlex 中文搜索无结果

**问题描述**：
使用中文关键词搜索 OpenAlex 返回 0 结果。

**解决方案**：
用 LLM 将中文查询转换为英文关键词：
```python
def extract_keywords(query: str) -> str:
    prompt = f"从以下中文问题中提取英文关键词：\n{query}"
    return llm.invoke(prompt)
```

---

### 四、文件处理问题

#### 6. 文件路径解析错误

**问题描述**：
上传文件名包含空格时，路径解析错误。

**解决方案**：
```python
# 错误
filename = user_input.split("upload ")[1]

# 正确
filename = user_input[len("upload "):].strip()
```

---

### 五、Streamlit 前端问题

#### 7. 页面刷新重复请求 API

**问题描述**：
每次 Streamlit 页面刷新都重复调用 `/api/files` 接口。

**解决方案**：
使用 session_state 缓存：
```python
def get_file_list():
    if st.session_state.get("files_loaded"):
        return
    # 请求数据...
    st.session_state.files_loaded = True
```

---

#### 8. 文件上传重复触发

**问题描述**：
`file_uploader` + `st.rerun()` 导致上传逻辑重复执行。

**解决方案**：
添加上传锁：
```python
if uploaded_file and not st.session_state.uploading:
    st.session_state.uploading = True
    # 上传逻辑...
    st.session_state.uploading = False
```

---

