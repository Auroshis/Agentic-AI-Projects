# gemini_client.py
from google import genai
import os


class GeminiClient:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return resp.text
