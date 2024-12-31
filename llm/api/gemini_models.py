import os
import time
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

from llm.base import *
from llm.format import *
from utils.global_functions import *

import google.generativeai as genai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import vertexai
GOOGLE_API_KEY = os.getenv(f"GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
PROJECT_ID = "gemini-infer"  # @param {type:"string"}
LOCATION = "us-central1"

class GeminiChat(LLMChat):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = GenerativeModel(self.model_name)
        
    def get_msg(self, messages: List[Message]) -> List[dict]:
        return [msg.to_gemini_format() for msg in messages]
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
                
        temperature = temperature if temperature else self.DEFAULT_TEMPERATURE
        max_tokens = max_tokens if max_tokens else self.DEFAULT_MAX_TOKENS
        num_comps = num_comps if num_comps else self.DEFAULT_NUM_COMPLETIONS
        inputs = self.get_msg(messages)

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        responses = self.model.generate_content(
            inputs,
            generation_config=GenerationConfig(
                candidate_count=num_comps,
                max_output_tokens=max_tokens,
                temperature=temperature),
        stream = False
        )

        for response in responses:
            return response.text
        
        time.sleep(self.DEFAULT_DELAY)
        
        return response.text
    