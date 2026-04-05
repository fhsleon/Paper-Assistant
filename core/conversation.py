"""对话逻辑：
1) 普通聊天：直接调用 LLM
2) 论文问答：调用 `ask_paper`
3) 图表问答：从已加载 PDF 提取图片 → vision_tools 识别 → LLM 结合识别结果回答
"""

from pathlib import Path
from parsers.pdf_parser import PDFParser
from config.prompts import SYSTEM_PROMPT
from config.utils import HISTORY_TURNS, response_text, UPLOAD_DIR


class ConversationMixin:

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def _recent_turns(self, thread_id: str):
        return list(self._history(thread_id)[-HISTORY_TURNS:])

    def _handle_search_papers(self, user_input: str, thread_id: str = "default") -> str:
        print(f"[工具调用] search_papers - 查询: {user_input[:50]}...")
        tool = self.tool_map.get("search_papers")
        if not tool:
            return "搜索工具不可用"
        result = tool.invoke({"query": user_input})
        print(f"[工具完成] search_papers")
        self._save_history(user_input, result, thread_id=thread_id)
        return result

    def _handle_general_chat(self, user_input: str, thread_id: str = "default") -> str:
        messages = (
            [{"role": "system", "content": self.SYSTEM_PROMPT}]
            + self._recent_turns(thread_id)
            + [{"role": "user", "content": user_input}]
        )
        content = response_text(self.llm.invoke(messages))
        self._save_history(user_input, content, thread_id=thread_id)
        return content

    def _handle_ask_paper(self, user_input: str, thread_id: str = "default") -> str:
        print(f"[工具调用] ask_paper - 问题: {user_input[:50]}...")
        tool = self.tool_map.get("ask_paper")
        if not tool:
            return "问答工具不可用"
        result = tool.invoke({"question": user_input})
        print(f"[工具完成] ask_paper")
        self._save_history(user_input, result, thread_id=thread_id)
        return result

    def _handle_analyze_image(self, user_input: str, thread_id: str = "default") -> str:
        """从已加载 PDF 提取图片 → vision_tools 识别 → LLM 结合识别结果回答。"""
        extract_tool = self.tool_map.get("extract_image_content")
        compare_tool = self.tool_map.get("compare_images")
        if not extract_tool and not compare_tool:
            return "图像分析工具不可用"

        paper_list = self.rag.get_paper_list()
        if not paper_list:
            return "没有已加载的论文，请先上传论文文件。"

        pdf_path = Path(paper_list[0]["path"])
        print(f"[视觉分析] 从 PDF 提取图片: {pdf_path.name}")

        parser = PDFParser(str(pdf_path))
        images = parser.extract_images(min_size=2000)

        if not images:
            return f"从论文 {pdf_path.name} 中未提取到有效图片（可能是扫描版或图片过小）。"

        print(f"[视觉分析] 提取到 {len(images)} 张图片")

        want_compare = any(k in user_input for k in ["对比", "比较", "vs", "versus"])

        if want_compare and compare_tool and len(images) >= 2:
            img1, img2 = images[0], images[1]
            print(f"[视觉工具] compare_images - {img1['filename']} vs {img2['filename']}")
            vision_result = compare_tool.invoke({"image_path1": img1["path"], "image_path2": img2["path"]})
        else:
            target_img = images[0]
            print(f"[视觉工具] extract_image_content - {target_img['filename']} (第{target_img['page']}页)")
            vision_result = extract_tool.invoke({"image_path": target_img["path"]})

        if len(images) > 1:
            extra_info = f"\n\n(注: 论文中共提取到 {len(images)} 张图片，当前分析了前 {'2' if want_compare else '1'} 张。如需分析其他图片请明确指出页码或图片序号)"
        else:
            extra_info = ""

        vision_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{user_input}\n\n图片识别结果：\n{vision_result}{extra_info}\n\n请基于识别结果回答用户问题；如果识别不充分，请说明并给出需要的补充信息。"},
        ]
        content = response_text(self.llm.invoke(vision_messages))
        self._save_history(user_input, content, thread_id=thread_id)
        return content
