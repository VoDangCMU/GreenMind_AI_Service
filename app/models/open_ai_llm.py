import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from typing import Optional

class OpenAILLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = "gpt-4.1-mini"

    def chat_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 512,
        temperature: float = 0
    ) -> str:

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        output_text = []

        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        output_text.append(c.text)

        return "\n".join(output_text).strip()


_openai_llm: Optional[OpenAILLM] = None

def get_openai_llm() -> OpenAILLM:
    global _openai_llm
    if _openai_llm is None:
        _openai_llm = OpenAILLM()
    return _openai_llm
