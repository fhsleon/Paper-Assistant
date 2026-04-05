from pathlib import Path
from langchain.tools import tool
from PIL import Image


def create_vision_tools(upload_dir: Path = None):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        model_path = "./model/Qwen2.5-VL-3B-Instruct"
        
        if not Path(model_path).exists():
            print(f"警告: 模型路径不存在: {model_path}")
            _model = None
            _processor = None
        else:
            print(f"正在加载视觉模型: {model_path}")
            _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            _model.eval()
            print("视觉模型加载完成")
    except Exception as e:
        print(f"加载视觉模型失败: {e}")
        _model = None
        _processor = None
    
    def _analyze_image(image_path: str, prompt: str) -> str:
        if _model is None or _processor is None:
            return "视觉模型未加载"
        
        try:
            import torch

            image = Image.open(image_path).convert('RGB')
            conversation = [
                {"role": "system", "content": "你是一个专业的论文图像分析助手。请用中文回答。"},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]

            text = _processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = _processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(_model.device)

            with torch.no_grad():
                output_ids = _model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            return _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        except Exception as e:
            return f"分析失败: {str(e)}"
    
    @tool
    def extract_image_content(image_path: str) -> str:
        """提取图片中的内容（图表、公式、表格等）。
        
        Args:
            image_path: 图片文件路径（支持 jpg, jpeg, png, gif, webp, bmp）
        
        使用场景：
        - 提取论文中的公式内容
        - 分析实验结果图表
        - 提取表格数据
        - 理解模型架构图
        """
        print(f"[视觉工具] extract_image_content - 分析图片: {image_path}")

        img_path = Path(image_path)
        if not img_path.exists() and upload_dir:
            img_path = upload_dir / image_path
        if not img_path.exists():
            return f"图片文件不存在: {image_path}"
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        if img_path.suffix.lower() not in valid_extensions:
            return f"不支持的图片格式: {img_path.suffix}"
        
        prompt = """请提取并描述这张图片中的内容：
1. 如果是公式：写出公式的LaTeX表示或数学表达式，并解释含义
2. 如果是图表：描述图表展示的数据、横纵坐标含义、主要结论
3. 如果是表格：提取表格内容（Markdown格式），并解释关键数据
4. 如果是架构图：描述模型结构、各模块功能、数据流向

请详细但简洁地回答。"""
        
        result = _analyze_image(str(img_path), prompt)
        print(f"[视觉工具] extract_image_content - 分析完成")
        return result
    
    @tool
    def compare_images(image_path1: str, image_path2: str) -> str:
        """对比两张图片（如对比实验结果）。
        
        Args:
            image_path1: 第一张图片路径
            image_path2: 第二张图片路径
        """
        print(f"[视觉工具] compare_images - 对比图片: {image_path1} vs {image_path2}")

        img1 = Path(image_path1)
        img2 = Path(image_path2)

        for img, name in [(img1, "第一张"), (img2, "第二张")]:
            if not img.exists() and upload_dir:
                img = upload_dir / img.name
            if not img.exists():
                return f"{name}图片不存在"
        
        if _model is None or _processor is None:
            return "视觉模型未加载"
        
        try:
            import torch

            image1 = Image.open(img1).convert('RGB')
            image2 = Image.open(img2).convert('RGB')
            conversation = [
                {"role": "system", "content": "你是一个专业的图像对比分析助手。请用中文回答。"},
                {"role": "user", "content": [
                    {"type": "text", "text": "请对比分析这两张图片。\n\n图片1："},
                    {"type": "image"},
                    {"type": "text", "text": "\n\n图片2："},
                    {"type": "image"},
                    {"type": "text", "text": "\n\n请分析：1. 主要区别 2. 各自的优缺点 3. 可以得出的结论"}
                ]}
            ]

            text = _processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = _processor(text=[text], images=[image1, image2], padding=True, return_tensors="pt")
            inputs = inputs.to(_model.device)

            with torch.no_grad():
                output_ids = _model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=True)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            result = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"[视觉工具] compare_images - 对比完成")
            return result
            
        except Exception as e:
            return f"对比失败: {str(e)}"
    
    return [extract_image_content, compare_images]
