"""OpenAlex 搜索、ask_paper、上传/列出/切换论文等 LangChain @tool。"""
import os
import re
import shutil
import requests
from pathlib import Path
from typing import Optional, Union, List

from langchain.tools import tool


def search_openalex(title: str, time: str = None, limit_page: int = 6, sort_by_date: bool = True) -> list:
    """调用 OpenAlex API 搜索论文。"""
    from datetime import datetime
    
    if time is None:
        time = str(datetime.now().year)
    
    url = "https://api.openalex.org/works"
    params = {
        "search": title,
        "filter": f"language:en|zh,publication_year:{time}",
        "per-page": limit_page,
        "sort": "publication_date:desc" if sort_by_date else "cited_by_count:desc",
        "mailto": os.environ.get('OPENALEX_EMAIL')
    }
    params = {k: v for k, v in params.items() if v}
    
    try:
        print(f"[DEBUG] OpenAlex 搜索: {title} (年份: {time})")
        response = requests.get(url, params=params, timeout=15)
        results = response.json().get("results", [])
        print(f"[DEBUG] 返回 {len(results)} 篇论文")
        return results
    except Exception as e:
        print(f"[DEBUG] 搜索失败: {e}")
        return [{"error": f"搜索失败: {e}"}]


def parse_scope_from_query(query: str, paper_count: int) -> Union[str, List[int]]:
    """从用户问题解析检索范围：all / current / 第几篇 → 下标。"""
    if not paper_count:
        return "current"
    
    query_lower = query.lower()
    
    if any(k in query_lower for k in ["所有", "全部", "这几篇", "all"]):
        return "all"
    if any(k in query_lower for k in ["这篇", "当前", "this", "current"]):
        return "current"
    
    matches = re.findall(r'第(\d+)篇', query)
    if matches:
        indices = [int(m) - 1 for m in matches if 0 <= int(m) - 1 < paper_count]
        return indices[0] if len(indices) == 1 else indices if indices else "current"
    
    return "current"


def create_paper_tools(rag, upload_dir: Path, llm=None):
    
    def extract_keywords(query: str) -> str:
        """用 LLM 从中文查询中提取英文关键词。"""
        if llm is None:
            return query
        
        prompt = f"""请从以下中文问题中提取出用于学术搜索的英文关键词，只输出关键词，用空格分隔，不要其他内容。

中文问题：{query}

英文关键词："""
        try:
            response = llm.invoke(prompt)
            keywords = str(response.content if hasattr(response, 'content') else response).strip().split('\n')[0]
            print(f"[关键词提取] '{query}' -> '{keywords}'")
            return keywords or query
        except Exception as e:
            print(f"[关键词提取失败] {e}")
            return query

    @tool
    def search_papers(query: str, year: Optional[str] = None) -> str:
        """搜索学术论文。当用户想查找、搜索论文时使用。
        
        Args:
            query: 搜索关键词(英文效果更好)
            year: 发表年份，可选
        """
        print(f"[工具调用] search_papers - 原始查询: {query}")
        results = search_openalex(extract_keywords(query), time=year)
        
        if not results or "error" in results[0]:
            return results[0].get("error", "未找到相关论文") if results else "未找到相关论文"
        
        papers = []
        for p in results[:4]:
            authors = ', '.join(a['author']['display_name'] for a in p.get('authorships', [])[:2])
            
            abstract = p.get('abstract_inverted_index', {})
            if abstract:
                words = sorted([(w, pos[0]) for w, pos in abstract.items()], key=lambda x: x[1])
                abstract_text = ' '.join(w for w, _ in words[:150]) + ("..." if len(words) > 150 else "")
            else:
                abstract_text = "无摘要"
            
            papers.append(
                f"【{p.get('title', 'N/A')}】\n"
                f"  作者: {authors}\n"
                f"  引用: {p.get('cited_by_count', 0)} | 年份: {p.get('publication_year', 'N/A')}\n"
                f"  摘要: {abstract_text}"
            )
        
        print(f"[工具完成] search_papers - 找到 {len(papers)} 篇")
        return "\n\n".join(papers)
    
    @tool
    def ask_paper(question: str) -> str:
        """对已上传的论文进行问答。这是最重要的工具！当用户询问论文的任何内容时都必须使用此工具。
        
        适用场景：
        - 用户询问论文的方法、模型、实验、结果、贡献等
        - 用户想了解论文的具体内容
        - 用户提问关于论文的任何问题
        
        Args:
            question: 关于论文的问题
        """
        files = list(upload_dir.glob("*.pdf"))
        if not files:
            return "❌ 没有上传任何论文，请先上传论文文件。"
        
        paper_count = len(rag.get_paper_list())
        
        if not rag.is_loaded():
            print(f"[ask_paper] 正在加载论文: {files[0].name}")
            load_result = rag.load_paper(str(files[0]))
            if not load_result.get("success"):
                return f"❌ 加载论文失败: {load_result.get('message', '未知错误')}"
        
        scope = parse_scope_from_query(question, paper_count)
        print(f"[ask_paper] 范围: {scope}, 问题: {question[:50]}...")
        
        result = rag.answer(question, scope=scope)
        return result["answer"] if result.get("success") else result.get("message", "问答失败")
    
    @tool
    def upload_paper(file_path: str) -> str:
        """上传PDF论文文件。
        
        Args:
            file_path: PDF文件的完整路径
        """
        src = Path(file_path)
        if not src.exists():
            return f"文件不存在: {file_path}"
        shutil.copy(src, upload_dir / src.name)
        return f"上传成功: {src.name}"
    
    @tool
    def list_files() -> str:
        """列出已上传的论文文件。当用户想知道有哪些论文时使用。"""
        print(f"[工具调用] list_files")
        files = [f.name for f in upload_dir.glob("*.pdf")]
        paper_list = rag.get_paper_list()
        
        if not files:
            return "暂无上传的论文文件"
        
        lines = []
        for i, f in enumerate(files, 1):
            loaded = next((p for p in paper_list if p["name"] == f), None)
            status = f" [已加载-第{loaded['idx'] + 1}篇]" if loaded else ""
            lines.append(f"  {i}. {f}{status}")
        
        print(f"[工具完成] list_files - {len(files)} 个文件")
        return "已上传的论文文件:\n" + "\n".join(lines)
    
    @tool
    def switch_paper(filename: str) -> str:
        """切换到指定的论文进行问答。当用户想切换到另一篇论文时使用。
        
        Args:
            filename: 论文文件名
        """
        print(f"[工具调用] switch_paper - {filename}")
        target = upload_dir / filename
        if not target.exists():
            return f"文件不存在: {filename}。可用: {[f.name for f in upload_dir.glob('*.pdf')]}"
        
        paper_list = rag.get_paper_list()
        existing = next((p for p in paper_list if p["name"] == filename), None)
        
        if existing:
            return f"论文 {filename} 已加载 (第{existing['idx'] + 1}篇)"
        
        result = rag.load_paper(str(target))
        if result.get("success"):
            idx = result.get("paper_idx", len(rag.get_paper_list()) - 1)
            return f"已加载论文: {filename} (第{idx + 1}篇)"
        return f"加载失败: {result.get('message', '未知错误')}"
    
    return [search_papers, ask_paper, upload_paper, list_files, switch_paper]
