import asyncio
from typing import List, Dict, Any, Optional, Iterator

from hydra import compose, initialize_config_dir
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, TaskType
import numpy as np

from utils.model_processor_audio_engine_factory import model_processor_audio_engine_factory
from utils.transcriber import Transcriber


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

    async def initialize(self):
        """Asynchronously triggers lazy loading framework factories on application startup."""
        print("⚡ [Service Startup] Initializing Hydra configurations and Whisper weights...")
        
        # Safe configuration loading targeting your explicit utils/ directory structure
        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            cfg = compose(config_name="config.yaml")
            
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