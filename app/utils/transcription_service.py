import asyncio
from typing import List, Dict, Any, Optional, Iterator
import re
import json

from hydra import compose, initialize_config_dir
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, TaskType
import numpy as np
from redis import asyncio as aioredis

from utils import model_processor_audio_engine_factory
from utils import Transcriber
from utils import SummarizationEngine


class TranscriptionService:
    """
    Encapsulates lifecycle, instantiation parameters, and runtime sessions 
    for Whisper processing networks. Decouples ML pipelines from FastAPI.
    """
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.model = None
        self.processor = None
        self.audio_engine = None
        self.transcriber = None
        self.redis: aioredis.Redis = None
        self.summarizer: Optional[SummarizationEngine] = None

    async def initialize(self):
        """Asynchronously triggers lazy loading framework factories on application startup."""
        print("⚡ [Service Startup] Initializing Hydra configurations and Whisper weights...")
        
        # Safe configuration loading targeting your explicit utils/ directory structure
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

        self.redis = aioredis.Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            db=cfg.redis.db,
            decode_responses=cfg.redis.decode_responses
        )

        self.summarizer = SummarizationEngine(cfg=cfg)
        await self.summarizer.initialize()
            
        task_type = getattr(TaskType, cfg.lora_config.task_type)
        lora_config = LoraConfig(
            r=cfg.lora_config.r,
            lora_alpha=cfg.lora_config.lora_alpha,
            target_modules=cfg.lora_config.target_modules,
            lora_dropout=cfg.lora_config.lora_dropout,
            bias=cfg.lora_config.bias,
            task_type=task_type
        )
        pretrained_model = WhisperForConditionalGeneration.from_pretrained(cfg.pretianed_whisper_path)
        model_config = dict(
            whisper_for_conditional_generation_model=pretrained_model,
            learning_rate=cfg.learning_rate,
            lora_config=lora_config
        )

        self.model, self.processor, self.audio_engine = model_processor_audio_engine_factory(
            pretianed_whisper_path=cfg.pretianed_whisper_path,
            model_config=model_config
        )

        self.transcriber = Transcriber(
            whisper_model=self.model,
            audio_engine=self.audio_engine
        )

    def run_offline(self, file_path: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Invokes sequential file reading iterations."""
        if not self.transcriber:
            raise RuntimeError("TranscriptionService has not been initialized.")
        return self.transcriber.transcribe_file(file_path, **kwargs)

    def create_stream_session(self, silence_timeout: float = 0.6, **kwargs):
        """Creates an isolated real-time streaming workspace container handle."""
        if not self.transcriber:
            raise RuntimeError("TranscriptionService has not been initialized.")
        return self.transcriber.stream_session(silence_timeout_s=silence_timeout, **kwargs)

    async def close(self, timeout_s: float = 5.0):
        """Drains outstanding execution queue tasks gracefully during framework teardown."""
        if self.transcriber:
            await self.transcriber.close(timeout_s=timeout_s)
        if self.redis:
            await self.redis.close()
            print("Redis connection cluster closed cleanly.")


    async def save_segment(self, session_id: str, segment: Dict[str, Any]):
        """
        Appends a newly finalized Whisper segment into Redis with an automatic 
        rolling 2-hour memory expiration (TTL) window.
        """
        raw_data = await self.redis.get(session_id)
        segments = json.loads(raw_data) if raw_data else []
        
        segments.append(segment)
        
        await self.redis.setex(name=session_id, time=7200, value=json.dumps(segments))


    async def search_transcript(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Queries Redis distributed node, executes deep case-insensitive evaluations,
        and derives localized character offset configurations.
        """
        raw_data = await self.redis.get(session_id)
        
        if raw_data is None:
            raise KeyError(f"Session {session_id} not found in the live cluster memory.")

        segments = json.loads(raw_data)
        results = []
        clean_query = query.strip().lower()
        
        for index, segment in enumerate(segments):
            text = segment.get("text", "")
            text_lower = text.lower()
            
            if clean_query in text_lower:
                offsets = [m.start() for m in re.finditer(re.escape(clean_query), text_lower)]
                
                results.append({
                    "segment_index": index,
                    "start_time": segment.get("start"),
                    "end_time": segment.get("end"),
                    "text": text,
                    "match_offsets": offsets
                })
                
        return results

    async def summarize_transcript(self, session_id: str, format_type: str) -> str:
        """Fetches complete context history lines from Redis cache and forwards data to the NLP Engine."""
        raw_data = await self.redis.get(session_id)
        if raw_data is None:
            raise KeyError(f"Session {session_id} not found in the live cluster memory.")

        segments = json.loads(raw_data)
        if not segments:
            return "No text contents found inside database cache storage to execute an evaluation summary."

        # Bind chronological array text segments into an unified single text payload context block
        full_conversation_text = " ".join([seg.get("text", "").strip() for seg in segments])
        
        if not self.summarizer:
            raise RuntimeError("The abstractive NLP processing summary pipeline has not been properly initialized.")

        # Dispatch the combined string to the Strategy Adapter for execution parsing
        return await self.summarizer.generate_summary(text=full_conversation_text, format_type=format_type)