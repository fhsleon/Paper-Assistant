"""路径、历史窗口、LLM 文本解析。"""
from pathlib import Path

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

HISTORY_TURNS = 32


def response_text(resp) -> str:
    if hasattr(resp, "content"):
        return str(resp.content)
    return str(resp)
