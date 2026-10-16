from typing import Any, List, Dict, Optional

import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor


class UnifiedAudioEngine:
    def __init__(
        self,
        processor: WhisperProcessor = None,
        sample_rate: int = 16000,
        max_audio_duration_seconds: float = 30.0,
        vad_repo: str = 'snakers4/silero-vad',
        vad_model: str = 'silero_vad',
        vad_trust_repo: bool = True
        ):
        if processor is None:
            raise AttributeError("Whisper processor is not provided!")
        self.processor = processor

        self.sample_rate = sample_rate
        self.max_duration = max_audio_duration_seconds
        self.max_samples = int(max_audio_duration_seconds * sample_rate)

        self.vad_model, utils = torch.hub.load(
            repo_or_dir=vad_repo,
            model=vad_model,
            trust_repo=vad_trust_repo
        )

        self.get_speech_timestamps, _, _, self.vad_iterator_class, _ = utils

        self.stream_buffer = torch.zeros(0, dtype=torch.float32)
        self.vad_iterator = self.vad_iterator_class(
            self.vad_model,
            sampling_rate=self.sample_rate
        )

    
    def reset_stream_state(self) -> None:
        self.stream_buffer = torch.zeros(0, dtype=torch.float32)
        self.vad_iterator = self.vad_iterator_class(
            self.vad_model,
            sampling_rate=self.sample_rate
        )

    def load_and_resample(self, audio_input: Any) -> torch.Tensor:
        if isinstance(audio_input, str):
            waveform, sr = torchaudio.load(audio_input)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            return waveform.squeeze(0)
            
        elif isinstance(audio_input, bytes):
            return torch.from_numpy(np.frombuffer(
                audio_input,
                dtype=np.int16
            ).copy()).float() / 32768.0
            
        elif isinstance(audio_input, torch.Tensor):
            return audio_input.float()
            
        elif isinstance(audio_input, np.ndarray):
            return torch.from_numpy(audio_input).float()
            
        else:
            raise ValueError(f"Unsupported audio input format: {type(audio_input)}")


    def chunk_offline_file(self, audio_input: Any) -> List[Dict[str, Any]]:
        """
        Case 1: Offline Processing.
        Splits a full file track into sub-30s pieces based on neural vocal boundaries.
        """
        waveform = self.load_and_resample(audio_input)
        
        if len(waveform) <= self.max_samples:
            return [{"array": waveform.numpy(), "start": 0.0, "end": len(waveform) / self.sample_rate}]

        speech_timestamps = self.get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=self.sample_rate
        )
        chunks = []
        if not speech_timestamps:
            return chunks

        current_chunk = []
        current_samples = 0
        chunk_start = speech_timestamps[0]['start'] / self.sample_rate

        for i, segment in enumerate(speech_timestamps):
            start, end = segment['start'], segment['end']
            duration = end - start

            if duration > self.max_samples:
                for idx in range(start, end, self.max_samples):
                    step_end = min(idx + self.max_samples, end)
                    chunks.append({
                        "array": waveform[idx:step_end].numpy(),
                        "start": idx / self.sample_rate,
                        "end": step_end / self.sample_rate
                    })
                continue

            if current_samples + duration > self.max_samples:
                chunks.append({
                    "array": torch.cat(current_chunk).numpy(),
                    "start": chunk_start,
                    "end": speech_timestamps[i-1]['end'] / self.sample_rate
                })
                current_chunk = [waveform[start:end]]
                current_samples = duration
                chunk_start = start / self.sample_rate
            else:
                current_chunk.append(waveform[start:end])
                current_samples += duration

        if current_chunk:
            chunks.append({
                "array": torch.cat(current_chunk).numpy(),
                "start": chunk_start,
                "end": speech_timestamps[-1]['end'] / self.sample_rate
            })

        return chunks

    def process_stream_frame(
            self,
            frame_bytes: bytes,
            silence_timeout_s: float = 1.0
        ) -> Optional[np.ndarray]:
        """
        Case 2: Live Real-Time Streaming (WebSockets).
        Appends live audio frames and yields full segments only when the user finishes speaking.
        """
        new_samples = self.load_and_resample(frame_bytes)
        self.stream_buffer = torch.cat([self.stream_buffer, new_samples])
        
        if len(self.stream_buffer) < 512:
            return None
            
        speech_timestamps = self.get_speech_timestamps(
            self.stream_buffer,
            self.vad_model,
            sampling_rate=self.sample_rate
        )
        
        # Scenario A: Hard limit threshold hit (30 seconds window max)
        if len(self.stream_buffer) >= self.max_samples:
            ready_chunk = self.stream_buffer[:self.max_samples].numpy()
            self.stream_buffer = self.stream_buffer[self.max_samples:]
            return ready_chunk
            
        # Scenario B: Target natural speaking sentence pause threshold reached
        if speech_timestamps and len(self.stream_buffer) > 0:
            last_speech_end_sample = speech_timestamps[-1]['end']
            trailing_silence_samples = len(self.stream_buffer) - last_speech_end_sample
            silence_duration_s = trailing_silence_samples / self.sample_rate
            
            if silence_duration_s >= silence_timeout_s:
                ready_chunk = self.stream_buffer.numpy()
                self.stream_buffer = torch.zeros(0, dtype=torch.float32)
                return ready_chunk

        return None

    def extract_features(
            self,
            audio_array: np.ndarray,
            text_target: str = None    
        ) -> Dict[str, Any]:
        """
        Case 3: Training Datasets Map.
        Translates raw input dictionaries directly to features inside our dataset pipelines.
        """
        features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate
        ).input_features
        
        output = {"input_features": features}
        if text_target is not None:
            output["labels"] = self.processor.tokenizer(text_target).input_ids
            output["text"] = text_target
            
        return output




        

        





