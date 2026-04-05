import shutil
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.agent import PaperReActAgent

upload_dir = Path("./uploads")
upload_dir.mkdir(exist_ok=True)

agent_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    agent_instance = PaperReActAgent()
    yield


app = FastAPI(title="论文智能助手 API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), auto_load: bool = True):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF 文件")
    
    safe_name = Path(file.filename).name
    temp_path = upload_dir / f"temp_{uuid.uuid4().hex}.pdf"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        final_path = upload_dir / safe_name
        if final_path.exists():
            final_path.unlink()
        temp_path.rename(final_path)
        
        if auto_load:
            result = agent_instance.rag.load_paper(str(final_path))
            return {"success": True, "filename": safe_name, "loaded": result.get("success", False)}
        
        return {"success": True, "filename": safe_name, "loaded": False}
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(500, str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        return {"response": agent_instance.chat(req.message, thread_id=req.thread_id)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/files")
async def list_files():
    files = [f.name for f in upload_dir.glob("*.pdf")]
    paper_list = agent_instance.rag.get_paper_list()
    
    return {
        "files": [
            {"name": f, "loaded": any(p["name"] == f for p in paper_list)}
            for f in files
        ]
    }


@app.get("/api/history")
async def get_history(thread_id: str = "default"):
    history = agent_instance.get_history(thread_id=thread_id)
    return {
        "history": [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in history
        ]
    }


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    file_path = upload_dir / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    
    file_path.unlink()
    paper_list = agent_instance.rag.get_paper_list()
    existing = next((p for p in paper_list if p["name"] == filename), None)
    if existing:
        agent_instance.rag.unload(existing["idx"])
    
    return {"success": True}


@app.post("/api/load/{filename}")
async def load_paper(filename: str):
    file_path = upload_dir / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    
    result = agent_instance.rag.load_paper(str(file_path))
    if result.get("success"):
        return {"success": True}
    raise HTTPException(500, result.get("message", "加载失败"))


@app.post("/api/clear")
async def clear_all():
    return agent_instance.clear_all()


@app.post("/api/history/clear")
async def clear_history(thread_id: Optional[str] = None):
    agent_instance.clear_history(thread_id=thread_id)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
