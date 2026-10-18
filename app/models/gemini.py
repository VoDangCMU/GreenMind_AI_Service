import requests
import json
from functools import lru_cache
import itertools
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", "")).split(",") if k.strip()]

api_key_cycle = itertools.cycle(API_KEYS)


class GeminiChat:
    def __init__(self, endpoint: str = GEMINI_ENDPOINT):
        self.endpoint = endpoint
        self.current_key = next(api_key_cycle)

    def _make_headers(self, key):
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": key
        }

    def _send_request(self, payload):
        """Thử gửi request bằng nhiều API key"""
        tried_keys = []

        for _ in range(len(API_KEYS)):
            key = self.current_key
            tried_keys.append(key)

            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self._make_headers(key),
                    data=json.dumps(payload)
                )

                # Nếu HTTP error → raise
                resp.raise_for_status()

                # Thành công → trả kết quả
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code

                # Nếu bị rate limit → đổi key → thử lại
                if status == 429:
                    print(f"[WARN] API key bị hạn chế 429 → đổi key khác...")
                    self.current_key = next(api_key_cycle)
                    continue

                # Nếu lỗi khác → raise
                print(f"[ERROR] Lỗi API key {key}: {e}")
                self.current_key = next(api_key_cycle)
                continue

        # Nếu đã thử hết key mà vẫn fail
        raise Exception(f"Tất cả API key đều lỗi: {tried_keys}")

    def chat_raw(self, system_prompt: str, user_prompt: str, max_output_tokens: int = 256):
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens
            }
        }

        return self._send_request(payload)

    def chat(self, text: str, max_output_tokens: int = 128):
        """Chat đơn giản với system prompt cố định."""
        return self.chat_raw("Bạn là trợ lý hữu ích.", text, max_output_tokens=max_output_tokens)


@lru_cache(maxsize=1)
def get_gemini() -> GeminiChat:
    return GeminiChat()
