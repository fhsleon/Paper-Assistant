"""PDF 文本 + 正则抽取公式片段 + PyMuPDF 图片提取，供 RAG / 视觉分析使用。"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

from PyPDF2 import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

IMAGE_OUTPUT_DIR = Path("./uploads/pdf_images")
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PDFParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    def extract_all(self) -> Dict:
        logger.info(f"开始解析 PDF: {self.pdf_path.name}")
        
        result = {
            "text": "",
            "formulas": [],
            "pages": 0
        }
        
        text_content = self._extract_text_pypdf()
        result["text"] = text_content["text"]
        result["pages"] = text_content["pages"]

        formulas = self._extract_formulas(result["text"])
        result["formulas"] = formulas
        
        logger.info(f"解析完成: {result['pages']} 页, {len(formulas)} 个公式")
        
        return result
    
    def _extract_text_pypdf(self) -> Dict:
        try:
            reader = PdfReader(str(self.pdf_path))
            pages = len(reader.pages)
            text = ""
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += f"\n\n=== 第 {i+1} 页 ===\n\n{page_text}"
            
            return {"text": text, "pages": pages}
        
        except Exception as e:
            logger.error(f"PyPDF2 提取失败: {e}")
            return {"text": "", "pages": 0}
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'-\n+', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _extract_formulas(self, text: str) -> List[Dict]:
        formulas = []
        
        latex_pattern = r'\$\$?([^\$]+)\$\$?'
        latex_matches = re.finditer(latex_pattern, text)
        
        for match in latex_matches:
            formula = match.group(1).strip()
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            formulas.append({
                "formula": formula,
                "context": context,
                "type": "latex"
            })
        
        math_symbols = r'[=+\-×÷∈⊙∑∏∫√≤≥≠±]'
        math_pattern = rf'[^\n]*{math_symbols}[^\n]*'
        math_matches = re.finditer(math_pattern, text)
        
        for match in math_matches:
            formula_line = match.group().strip()
            if not any(f["formula"] == formula_line for f in formulas):
                formulas.append({
                    "formula": formula_line,
                    "context": formula_line,
                    "type": "math_symbols"
                })
        
        logger.info(f"提取到 {len(formulas)} 个公式")
        return formulas
    
    def get_enhanced_text(self) -> str:
        result = self.extract_all()
        
        enhanced_parts = []
        
        enhanced_parts.append("=== 论文正文 ===\n")
        enhanced_parts.append(result["text"])
        
        if result["formulas"]:
            enhanced_parts.append("\n\n=== 数学公式 ===\n")
            for i, formula in enumerate(result["formulas"], 1):
                enhanced_parts.append(f"\n[公式 {i}] {formula['formula']}\n")
                if formula["type"] == "latex":
                    enhanced_parts.append(f"上下文: {formula['context']}\n")
        
        return "".join(enhanced_parts)
    
    def save_enhanced_text(self, output_path: str) -> None:
        enhanced_text = self.get_enhanced_text()
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w", encoding="utf-8") as f:
            f.write(enhanced_text)
        
        logger.info(f"增强文本已保存到: {output}")

    def extract_images(self, min_size: int = 2000) -> List[Dict]:
        """用 PyMuPDF 从 PDF 中提取图片，返回图片信息列表。

        Args:
            min_size: 最小图片面积 (宽*高)，过滤小图标/装饰图

        Returns:
            [
                {"path": "uploads/pdf_images/xxx_page3_img0.png", "page": 3, "index": 0},
                ...
            ]
        """
        try:
            import fitz
        except ImportError:
            logger.error("需要安装 PyMuPDF: pip install PyMuPDF")
            return []

        pdf_name = self.pdf_path.stem
        out_dir = IMAGE_OUTPUT_DIR / pdf_name
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(self.pdf_path))
        images = []
        img_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if not base_image or base_image.get("width", 0) * base_image.get("height", 0) < min_size:
                    continue

                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                filename = f"page{page_num + 1}_img{img_index}.{ext}"
                filepath = out_dir / filename

                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "path": str(filepath),
                    "filename": filename,
                    "page": page_num + 1,
                    "index": img_index,
                    "width": base_image.get("width"),
                    "height": base_image.get("height")
                })
                img_index += 1

        doc.close()
        logger.info(f"从 {self.pdf_path.name} 提取了 {len(images)} 张图片 -> {out_dir}")
        return images


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pdf_parser.py <PDF路径> [输出路径]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "enhanced_text.txt"
    
    parser = PDFParser(pdf_path)
    parser.save_enhanced_text(output_path)
    
    result = parser.extract_all()
    print(f"\n总页数: {result['pages']}")
    print(f"公式数: {len(result['formulas'])}")
    print(f"文本长度: {len(result['text'])} 字符")
