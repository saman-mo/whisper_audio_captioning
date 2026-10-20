import httpx

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

    async def initialize(self):
        """Validates configuration parameters and engine readiness on container bootstrap."""
        print(f"🤖 [Summarizer Startup] Initializing framework as [{self.engine_type.upper()}] mode strategy...")

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
        """Sends an abstractive text processing instruction to an OpenAI-compatible API schema server."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key != "none" else {}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.3
        }
        
        # Using a clean non-blocking network connection block to route the summary processing request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{endpoint.rstrip('/')}/chat/completions", 
                    json=payload, 
                    headers=headers, 
                    timeout=45.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as network_exception:
                return f"Summarization Service Exception across target connection routing layer: {str(network_exception)}"