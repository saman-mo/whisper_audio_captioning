from typing import Dict

from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

from models.model import AudioTranscriber
from audio_engine import UnifiedAudioEngine


def model_processor_audio_engine_factory(
        pretianed_whisper_path: str = None,
        fine_tuned_whisper_wts: str = None,
        model_config: Dict = None,
        device: str = 'cpu',
        sample_rate: int = 16000,
        max_audio_duration_seconds: float = 30.0
    ) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    
    if not pretianed_whisper_path:
        raise ValueError("pretianed_whisper_path should be provided!")
    
    if model_config:   
        model = AudioTranscriber(**model_config)
        if fine_tuned_whisper_wts:
            model.load_state_dict(torch.load(fine_tuned_whisper_wts, map_location=device))
    else:
        model = WhisperForConditionalGeneration.from_pretrained(pretianed_whisper_path)

    processor = WhisperProcessor.from_pretrained(pretianed_whisper_path)

    audio_engine = UnifiedAudioEngine(
        processor=processor,
        sample_rate=sample_rate,
        max_audio_duration_seconds=max_audio_duration_seconds
    )

    return model, processor, audio_engine
            



        
    