from config.prompts import INTENT_PROMPT
from config.utils import response_text
import re


class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, user_input: str) -> str:
        prompt = INTENT_PROMPT.format(user_input=user_input)
        raw_response = response_text(self.llm.invoke(prompt))
        
        first_line = raw_response.strip().split('\n')[0].strip()
        print(f"[DEBUG] 模型输出首行: {first_line}")
        
        valid_intents = ["SEARCH_PAPERS", "ASK_PAPER", "ANALYZE_IMAGE", "GENERAL_CHAT"]
        
        for name in valid_intents:
            if re.search(rf'\b{name}\b', first_line, re.IGNORECASE):
                return name
        
        return "GENERAL_CHAT"
