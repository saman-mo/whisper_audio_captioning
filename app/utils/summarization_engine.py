import httpx
import asyncio
import random


class InferenceEngineException(Exception):
    """Custom exception raised when the backend inference server fails catastrophically."""
    pass



class SummarizationEngine:
    """
    Decouples text summarization engines from the primary Whisper pipeline.
    Utilizes an async HTTP client strategy to process requests through a local 
    decoupled server framework (vLLM/Ollama) or external public platforms.
    """
    def __init__(self, cfg):
        self.engine_type = cfg.summarizer.engine_type
        self.local_api_endpoint = cfg.summarizer.local_api_endpoint
        self.local_model_name = cfg.summarizer.local_model_name
        
        self.external_api_endpoint = cfg.summarizer.external_api_endpoint
        self.external_api_key = cfg.summarizer.external_api_key
        self.external_model_name = cfg.summarizer.external_model_name
        
        self.max_tokens = cfg.summarizer.max_tokens

        self.temperature = cfg.summarizer.temperature
        self.max_retries = cfg.summarizer.max_retries
        self.initial_backoff = cfg.summarizer.initial_backoff_seconds
        self.timeout = cfg.summarizer.timeout_seconds

    async def initialize(self):
        """Validates configuration parameters and engine readiness on container bootstrap."""
        print(f"[Summarizer Startup] Initializing framework as [{self.engine_type.upper()}] mode strategy...")

    async def generate_summary(self, text: str, format_type: str) -> str:
        """Processes raw text collections and maps execution paths based on current strategies."""
        prompt = (
            f"You are an elite enterprise assistant. Analyze the following conversation transcript data "
            f"and generate an executive summary formatted strictly as {format_type}. "
            f"Do not introduce introductory or conversational filler text. Return only the summary text:\n\n{text}"
        )
        
        if self.engine_type == "local":
            return await self._dispatch_request(
                endpoint=self.local_api_endpoint,
                model_name=self.local_model_name,
                prompt=prompt,
                api_key="none"
            )
        elif self.engine_type == "external_api":
            return await self._dispatch_request(
                endpoint=self.external_api_endpoint,
                model_name=self.external_model_name,
                prompt=prompt,
                api_key=self.external_api_key
            )
        else:
            raise ValueError(f"Invalid engine configuration state: {self.engine_type}")

    async def _dispatch_request(self, endpoint: str, model_name: str, prompt: str, api_key: str) -> str:
        """
        Sends an abstractive text processing instruction to an OpenAI-compatible API schema server.
        Implements transient error retries with exponential backoff and random jitter.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key != "none" else {}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        url = f"{endpoint.rstrip('/')}/chat/completions"
        
        async with httpx.AsyncClient() as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.post(
                        url, 
                        json=payload, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                    
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    print(f"[Attempt {attempt}/{self.max_retries}] Inference connection issue encountered: {str(exc)}")
                    
                    if attempt == self.max_retries:
                        raise InferenceEngineException(
                            f"Inference cluster routing layer failed completely after {self.max_retries} attempts. Underlying error: {str(exc)}"
                        )
                    
                    # Compute dynamic exponential backoff time with random jitter using parameterized configs
                    sleep_time = (self.initial_backoff * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                    print(f"Sleeping for {sleep_time:.2f}s before execution re-attempt...")
                    await asyncio.sleep(sleep_time)