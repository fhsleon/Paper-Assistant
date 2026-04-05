"""RAG：BM25 + 向量 + RRF 融合，可选 CrossEncoder 重排；多论文统一索引。"""
import os
import json
import hashlib
import logging
import pickle
from typing import Optional, List, Union, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from parsers.pdf_parser import PDFParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


RAG_PROMPT = """
### 角色定义
你是一个专业的论文问答助手，基于用户上传的论文内容进行回答问题。

### 任务描述
1. 根据用户提出的问题进行解析，提取关键词。
2. 根据提取的关键词，从论文内容中检索相关段落。
3. 根据检索到的段落，生成专业的回答。

### 注意事项
1. 回答的内容必须出自论文，不能编造信息。
2. 尽可能从论文中提取相关信息回答，即使信息不完整也要给出部分答案。只有在论文中完全没有相关信息时才回答"无法回答"。
3. 尽量使用论文原文的专业术语和表述，关键数值、方法名称、模块名称必须精确引用。
4. 数学公式使用LaTeX格式（$...$）。
5. 回答完所有问题后，标注信息来源，如：[来自论文1] xxx、[来自论文2] xxx。
6. 如果论文内容模糊或矛盾，说明情况并提供多个可能的解释。

论文内容:
{context}

问题:
{question}
"""


class RAGConfig:
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", "!", "?", ";", ",", " ", ""]
    
    BM25_K = 5
    VECTOR_K = 5
    RRF_K = 60
    
    RERANK_TOP_K = 5
    RERANKER_MAX_LENGTH = 512
    
    CONTEXT_MAX_LENGTH = 5000
    
    CACHE_DIR = "./vector_cache"


class PaperRAG:
    def __init__(
        self, 
        rag_llm, 
        embeddings, 
        reranker_path: Optional[str] = None, 
        reranker_device: str = "cpu",
        config: Optional[RAGConfig] = None
    ):
        self.llm = rag_llm
        self.embeddings = embeddings
        self.config = config or RAGConfig()
        
        self.vectorstore = None
        self.bm25_retriever = None
        self.vector_retriever = None
        self.reranker = None
        self.reranker_device = reranker_device
        self.reranker_path = reranker_path
        
        self.papers: List[Dict[str, Any]] = []
        self.all_chunks: List[Document] = []
        
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=self.config.SEPARATORS,
        )
    
    def _get_cache_key(self, file_path: str) -> str:
        stat = os.stat(file_path)
        return hashlib.md5(f"{file_path}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, str]:
        base = os.path.join(self.cache_dir, cache_key)
        return {
            "vector": f"{base}_vector",
            "bm25": f"{base}_bm25.pkl",
            "chunks": f"{base}_chunks.pkl",
            "meta": f"{base}_meta.json",
        }
    
    def _save_cache(self, cache_key: str, paper_name: str, chunks: List[Document]):
        paths = self._get_cache_paths(cache_key)
        try:
            self.vectorstore.save_local(paths["vector"])
            pickle.dump(self.bm25_retriever, open(paths["bm25"], "wb"))
            pickle.dump(chunks, open(paths["chunks"], "wb"))
            json.dump({"paper_name": paper_name, "chunks_count": len(chunks)}, 
                      open(paths["meta"], "w", encoding="utf-8"), ensure_ascii=False)
            logger.info(f"缓存已保存: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def _load_cache(self, cache_key: str) -> Tuple[bool, List[Document]]:
        paths = self._get_cache_paths(cache_key)
        if not all(os.path.exists(p) for p in paths.values()):
            return False, []
        
        try:
            self.vectorstore = FAISS.load_local(paths["vector"], self.embeddings, allow_dangerous_deserialization=True)
            self.bm25_retriever = pickle.load(open(paths["bm25"], "rb"))
            chunks = pickle.load(open(paths["chunks"], "rb"))
            self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.VECTOR_K})
            
            meta = json.load(open(paths["meta"], encoding="utf-8"))
            logger.info(f"从缓存加载: {meta['paper_name']} ({meta['chunks_count']} 个块)")
            return True, chunks
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return False, []
    
    def load_paper(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"success": False, "message": f"文件不存在: {file_path}"}
        
        try:
            logger.info(f"加载论文: {file_path}")
            paper_name = os.path.basename(file_path)
            paper_idx = len(self.papers)
            cache_key = self._get_cache_key(file_path)
            
            cache_loaded, cached_chunks = self._load_cache(cache_key)
            if cache_loaded:
                for chunk in cached_chunks:
                    chunk.metadata["paper_idx"] = paper_idx
                self.all_chunks.extend(cached_chunks)
                
                if self.vectorstore:
                    cached_vs = FAISS.load_local(self._get_cache_paths(cache_key)["vector"], 
                                                  self.embeddings, allow_dangerous_deserialization=True)
                    self.vectorstore.merge_from(cached_vs)
                
                self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                self.bm25_retriever.k = self.config.BM25_K
                self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.VECTOR_K})
                
                self.papers.append({"name": paper_name, "path": file_path, "idx": paper_idx, 
                                    "cache_key": cache_key, "from_cache": True})
                logger.info(f"论文从缓存加载成功: {paper_name}")
                return {"success": True, "message": f"从缓存加载: {paper_name}", 
                        "paper_idx": paper_idx, "paper_name": paper_name, "from_cache": True}
            
            parser = PDFParser(file_path)
            enhanced_result = parser.extract_all()
            
            docs = PyPDFLoader(file_path).load()
            for doc in docs:
                doc.page_content = PDFParser.clean_text(doc.page_content)
            
            formula_docs = [
                Document(
                    page_content=f"数学公式: {f['formula']}\n上下文: {f['context']}",
                    metadata={"source": file_path, "type": "formula", "paper_name": paper_name, "paper_idx": paper_idx}
                ) for f in enhanced_result["formulas"]
            ]
            
            text_chunks = self.splitter.split_documents(docs)
            for chunk in text_chunks:
                chunk.metadata.update({"paper_name": paper_name, "paper_idx": paper_idx, "type": "text"})
            
            paper_chunks = text_chunks + formula_docs
            self.all_chunks.extend(paper_chunks)
            self._rebuild_index()
            self._save_cache(cache_key, paper_name, paper_chunks)
            
            self.papers.append({
                "name": paper_name, "path": file_path, "idx": paper_idx,
                "pages": len(docs), "chunks": len(text_chunks), "formulas": len(formula_docs),
                "cache_key": cache_key, "from_cache": False
            })
            
            logger.info(f"论文加载成功: {paper_name} (索引 {paper_idx})")
            return {"success": True, "message": f"加载成功: {paper_name}",
                    "paper_idx": paper_idx, "paper_name": paper_name, "from_cache": False}
            
        except Exception as e:
            logger.error(f"加载失败: {e}")
            return {"success": False, "message": f"加载失败: {str(e)}"}
    
    def _rebuild_index(self):
        if not self.all_chunks:
            self.vectorstore = None
            self.bm25_retriever = None
            self.vector_retriever = None
            return
        
        self.vectorstore = FAISS.from_documents(self.all_chunks, self.embeddings)
        
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        self.bm25_retriever.k = self.config.BM25_K
        
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.VECTOR_K}
        )
        
        logger.info(f"索引重建完成: {len(self.all_chunks)} 个块")
    
    def unload(self, paper_idx: Optional[int] = None):
        if paper_idx is None:
            self.papers = []
            self.all_chunks = []
            self._rebuild_index()
            logger.info("已卸载所有论文")
        else:
            self.papers = [p for p in self.papers if p["idx"] != paper_idx]
            self.all_chunks = [c for c in self.all_chunks if c.metadata.get("paper_idx") != paper_idx]
            for i, p in enumerate(self.papers):
                p["idx"] = i
            for c in self.all_chunks:
                old_idx = c.metadata.get("paper_idx", 0)
                new_idx = next((i for i, p in enumerate(self.papers) if p["name"] == c.metadata.get("paper_name")), 0)
                c.metadata["paper_idx"] = new_idx
            self._rebuild_index()
            logger.info(f"已卸载论文索引 {paper_idx}")
    
    def is_loaded(self) -> bool:
        return len(self.papers) > 0 and self.bm25_retriever is not None

    def get_paper_list(self) -> List[Dict[str, Any]]:
        return self.papers

    def _rrf_fusion(self, bm25_docs: List, vector_docs: List) -> List:
        k = self.config.RRF_K
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_key = doc.page_content + str(doc.metadata.get("paper_idx"))
            if doc_key not in doc_scores:
                doc_scores[doc_key] = 0
                doc_map[doc_key] = doc
            doc_scores[doc_key] += 1 / (k + rank)
        
        for rank, doc in enumerate(vector_docs, start=1):
            doc_key = doc.page_content + str(doc.metadata.get("paper_idx"))
            if doc_key not in doc_scores:
                doc_scores[doc_key] = 0
                doc_map[doc_key] = doc
            doc_scores[doc_key] += 1 / (k + rank)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_key] for doc_key, _ in sorted_docs]
    
    def _filter_by_scope(self, docs: List, scope: Union[str, List[int]]) -> List:
        if scope == "all" or scope is None:
            return docs
        if scope == "current":
            if not self.papers:
                return []
            current_idx = self.papers[-1]["idx"]
            return [d for d in docs if d.metadata.get("paper_idx") == current_idx]
        if isinstance(scope, list):
            return [d for d in docs if d.metadata.get("paper_idx") in scope]
        return docs
    
    def _rerank(self, question: str, docs: List) -> List:
        top_k = self.config.RERANK_TOP_K
        
        if not docs:
            return docs[:top_k]
        
        if self.reranker is None and self.reranker_path and os.path.exists(self.reranker_path):
            try:
                self.reranker = CrossEncoder(
                    self.reranker_path, 
                    max_length=self.config.RERANKER_MAX_LENGTH, 
                    device=self.reranker_device
                )
                logger.info(f"重排序模型加载成功")
            except Exception as e:
                logger.warning(f"重排序模型加载失败: {e}")
                return docs[:top_k]
        
        if not self.reranker:
            return docs[:top_k]
        
        try:
            pairs = [[question, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        except Exception as e:
            logger.warning(f"重排序失败: {e}")
            return docs[:top_k]
    
    def retrieve(self, query: str, scope: Union[str, List[int]] = None) -> List[Document]:
        if not self.is_loaded():
            return []
        
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        docs = self._rrf_fusion(bm25_docs, vector_docs)
        docs = self._filter_by_scope(docs, scope)
        docs = self._rerank(query, docs)
        
        return docs
    
    def _format_context_with_source(self, docs: List[Document]) -> str:
        context_parts = []
        for doc in docs:
            paper_name = doc.metadata.get("paper_name", "未知论文")
            paper_idx = doc.metadata.get("paper_idx", 0)
            context_parts.append(f"[来自论文{paper_idx + 1}: {paper_name}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)[:self.config.CONTEXT_MAX_LENGTH]
    
    def answer(self, query: str, scope: Union[str, List[int]] = None) -> dict:
        if not self.is_loaded():
            return {"success": False, "answer": "", "message": "请先加载论文"}
        
        try:
            docs = self.retrieve(query, scope)
            
            if not docs:
                return {
                    "success": True,
                    "answer": "未找到相关内容",
                    "context": "",
                    "retrieved_docs": []
                }
            
            context = self._format_context_with_source(docs)
            prompt = RAG_PROMPT.format(context=context, question=query)
            
            print(f"[RAG] 正在生成回答，检索到 {len(docs)} 个文档片段...")
            response = self.llm.invoke(prompt)
            answer = getattr(response, 'content', str(response))
            print(f"[RAG] 回答生成完成")
            
            return {
                "success": True,
                "answer": answer,
                "context": context,
                "retrieved_docs": docs
            }
            
        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {"success": False, "answer": "", "message": f"问答失败: {str(e)}"}
    
    def run(self, question: str, file_path: Optional[str] = None) -> dict:
        if file_path:
            result = self.load_paper(file_path)
            if not result["success"]:
                return result
        
        return self.answer(question)
