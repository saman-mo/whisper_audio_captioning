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
        self.config = None

    async def initialize(self):
        """Asynchronously triggers lazy loading framework factories on application startup."""
        print("⚡ [Service Startup] Initializing Hydra configurations and Whisper weights...")
        
        # Safe configuration loading targeting your explicit utils/ directory structure
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")

        self.config = cfg

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
            audio_engine=self.audio_engine,
            config=self.config
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
        if self.summarizer:
            await self.summarizer.close()
        if self.redis:
            await self.redis.close()
            print("Redis connection cluster closed cleanly.")


    async def save_segment(self, session_id: str, segment: Dict[str, Any]):
        """
        Appends a newly finalized Whisper segment into Redis atomically using List primitives.
        Enforces a rolling 2-hour memory expiration (TTL) window on the session workspace key.
        """
        key = f"session:{session_id}"

        try:
            # Atomically push the serialized segment JSON onto the tail of the list
            await self.redis.rpush(key, json.dumps(segment))
            
            # Ensure the entire list maintains a rolling 2-hour expiration window from the last write
            await self.redis.expire(key, 7200)
        except aioredis.exceptions.ResponseError as redis_err:
            if "OOM" in str(redis_err):
                print(f"[Cluster Alert] Redis cache memory limits exceeded! Failed to write session: {session_id}")
                raise MemoryError("Cache cluster storage capacity reached. Deep transcript persistence failed.")
            raise redis_err



    async def search_transcript(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Queries Redis distributed node, executes deep case-insensitive evaluations,
        and derives localized character offset configurations.
        """
        key = f"session:{session_id}"
        
        # Retrieve all elements from the atomic list cleanly
        raw_elements = await self.redis.lrange(key, 0, -1)
        
        # Explicitly check if the key exists to catch missing or expired sessions correctly
        if not raw_elements and not await self.redis.exists(key):
            raise KeyError(f"Session {session_id} not found in the live cluster memory.")

        results = []
        clean_query = query.strip().lower()
        
        for index, raw_item in enumerate(raw_elements):
            segment = json.loads(raw_item)
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
        """Fetches the complete context history list from Redis cache and forwards data to the NLP Engine."""
        key = f"session:{session_id}"
        
        # Extract the chronological list arrays from the database segment
        raw_elements = await self.redis.lrange(key, 0, -1)
        
        if not raw_elements and not await self.redis.exists(key):
            raise KeyError(f"Session {session_id} not found in the live cluster memory.")

        if not raw_elements:
            return "No text contents found inside database cache storage to execute an evaluation summary."

        # Reconstruct the text collection sequentially from the atomic list items
        segments = [json.loads(item) for item in raw_elements]
        full_conversation_text = " ".join([seg.get("text", "").strip() for seg in segments])
        
        if not self.summarizer:
            raise RuntimeError("The abstractive NLP processing summary pipeline has not been properly initialized.")

        # Dispatch the combined string to the Strategy Adapter for execution parsing
        return await self.summarizer.generate_summary(text=full_conversation_text, format_type=format_type)