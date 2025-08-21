from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
import torch, os

DEFAULT_QWEN = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

class QwenChat:
    def __init__(self, model_id: str = DEFAULT_QWEN):
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        kwargs = dict(trust_remote_code=True)
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16   
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.config.use_cache = True

    def chat_raw(self, system_prompt: str, user_prompt: str, max_new_tokens=256):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([prompt], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = out[0, inputs["input_ids"].shape[1]:]           # chỉ phần sinh ra
        return self.tok.decode(gen_ids, skip_special_tokens=True)

    def chat(self, text: str, max_new_tokens=128):
        """Chat đơn giản: chỉ có system fixed + user input"""
        return self.chat_raw("Bạn là trợ lý hữu ích.", text, max_new_tokens=max_new_tokens)

@lru_cache(maxsize=1)
def get_qwen() -> QwenChat:
    return QwenChat()
