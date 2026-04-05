"""PaperReActAgent：加载模型与 RAG、意图路由、会话与文件操作。"""
import gc
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage

from core.rag import PaperRAG
from core.conversation import ConversationMixin
from core.intent_classifier import IntentClassifier
from config.utils import UPLOAD_DIR
from tools.paper_tools import create_paper_tools
from tools.vision_tools import create_vision_tools

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = os.environ.get('LANGCHAIN_ENDPOINT')
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')


class PaperReActAgent(ConversationMixin):
    def __init__(self):
        model_path = "./model/Qwen2.5-3B-Instruct"
        print(f"正在加载语言模型: {model_path}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("语言模型加载完成")
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            raise

        self.embeddings = HuggingFaceEmbeddings(
            model_name="./model/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.rag = PaperRAG(self.llm, self.embeddings, reranker_path="./model/bge-reranker-v2-m3")

        all_tools = create_paper_tools(self.rag, UPLOAD_DIR, llm=self.llm) + create_vision_tools(upload_dir=UPLOAD_DIR)
        self.tool_map = {t.name: t for t in all_tools}
        self.intent_classifier = IntentClassifier(self.llm)
        self._histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def _history(self, thread_id: str) -> List[Dict[str, str]]:
        return self._histories[thread_id]

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        return self._histories["default"]

    def get_history(self, thread_id: str = "default"):
        return [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in self._history(thread_id)
        ]

    def chat(self, user_input: str, thread_id: str = "default") -> str:
        intent = self.intent_classifier.classify(user_input)
        print(f"[意图识别] {intent}")

        if intent == "GENERAL_CHAT":
            return self._handle_general_chat(user_input, thread_id)

        if intent == "ASK_PAPER" and self._looks_like_chart_question(user_input) and self._has_chart_images():
            intent = "ANALYZE_IMAGE"

        handlers = {
            "ASK_PAPER": self._handle_ask_paper,
            "SEARCH_PAPERS": self._handle_search_papers,
            "ANALYZE_IMAGE": self._handle_analyze_image,
        }
        
        handler = handlers.get(intent, self._handle_general_chat)
        return handler(user_input, thread_id=thread_id)

    def _has_chart_images(self) -> bool:
        return any(UPLOAD_DIR.glob(f"*{ext}") for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"])

    @staticmethod
    def _looks_like_chart_question(text: str) -> bool:
        keywords = ["图", "图表", "表格", "公式", "对比", "比较", "vs", "指标", "mIoU", "mDice", "F1", "结果", "实验"]
        return any(k in text for k in keywords)

    def _save_history(self, user_input: str, response: str, thread_id: str = "default"):
        hist = self._histories[thread_id]
        hist.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": response}])
        if len(hist) > 100:
            self._histories[thread_id] = hist[-100:]

    def upload(self, file_path: str, auto_load: bool = True) -> dict:
        src = Path(file_path)
        if not src.exists():
            return {"success": False, "message": f"文件不存在: {file_path}"}

        dst = UPLOAD_DIR / src.name
        shutil.copy(src, dst)

        if not auto_load:
            return {"success": True, "filename": src.name, "loaded": False}

        load_result = self.rag.load_paper(str(dst))
        if load_result.get("success"):
            return {"success": True, "filename": src.name, "message": f"上传并加载成功: {src.name}", "loaded": True}
        return {"success": True, "filename": src.name, "message": f"上传成功，但加载失败: {load_result.get('message')}", "loaded": False}

    def list_files(self) -> list:
        return [f.name for f in UPLOAD_DIR.glob("*.pdf")]

    def unload(self, delete_file: bool = False) -> dict:
        self.rag.unload()
        result = {"success": True, "message": "已卸载论文"}

        if delete_file:
            for f in UPLOAD_DIR.glob("*.pdf"):
                f.unlink()
            result["deleted_files"] = True

        gc.collect()
        return result

    def clear_all(self, clear_history: bool = True) -> dict:
        papers_count = len(self.rag.get_paper_list()) if self.rag.is_loaded() else 0
        self.rag.unload()

        files_deleted = sum(1 for f in UPLOAD_DIR.glob("*.pdf") if (f.unlink() or True))

        if clear_history:
            self._histories.clear()

        gc.collect()
        return {
            "success": True,
            "message": f"已卸载 {papers_count} 篇论文，删除 {files_deleted} 个文件",
            "papers_unloaded": papers_count,
            "files_deleted": files_deleted,
            "history_cleared": clear_history
        }

    def clear_history(self, thread_id: Optional[str] = None):
        if thread_id is None:
            self._histories.clear()
        else:
            self._histories[thread_id] = []
