from typing import Iterator, Dict, Any, Optional, List
import asyncio

import torch
import numpy as np

from models.model import AudioTranscriber
from utils.audio_engine import UnifiedAudioEngine


class BatchRequest:
    def __init__(self, audio_array: np.ndarray, prefix_tokens: Optional[List[int]], generation_kwargs: dict):
        self.audio_array = audio_array
        self.prefix_tokens = prefix_tokens
        self.generation_kwargs = generation_kwargs
        
        self.event = asyncio.Event()
        self.result_text: str = ""
        self.result_tokens: List[int] = []


class AsyncStreamContext:
    """
    Stateful streaming manager that tracks timeline offsets, handles raw wave carryover stitching,
    and coordinates dynamic batch queue job submissions for a single connection lifeline.
    """
    def __init__(
            self,
            transcriber: Any = None,
            silence_timeout_s: float = 1.0,
            max_context_tokens: int = 150,
            overlap_seconds: float = 0.5,
            generation_kwargs: dict = None
    ):
        if transcriber is None:
            raise AttributeError("Transcriber is not provided!")
        self.tx = transcriber
        self.timeout = silence_timeout_s
        self.max_tokens = max_context_tokens
        self.overlap_samples = int(overlap_seconds * transcriber.audio_engine.sample_rate)
        self.generation_kwargs = generation_kwargs
        
        # Isolated runtime state variables per active WebSocket connection
        self.stream_time_offset = 0.0
        self.flushed_result: Optional[Dict[str, Any]] = None
        self.token_history: List[int] = []
        self.audio_carryover: Optional[np.ndarray] = None
        
    def __enter__(self):
        self.tx.audio_engine.reset_stream_state()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tx.audio_engine.reset_stream_state()

    async def process_frame(self, frame_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Consumes incoming frame buffers, applies raw overlap signals, and hits the batch queue."""
        ready_chunk = self.tx.audio_engine.process_stream_frame(frame_bytes, silence_timeout_s=self.timeout)
        
        if ready_chunk is not None:
            chunk_duration = len(ready_chunk) / self.tx.audio_engine.sample_rate
            start_time = self.stream_time_offset
            end_time = start_time + chunk_duration
            self.stream_time_offset = end_time
            
            processing_audio = ready_chunk
            if self.audio_carryover is not None:
                processing_audio = np.concatenate([self.audio_carryover, processing_audio])
            
            if len(ready_chunk) >= self.overlap_samples:
                self.audio_carryover = ready_chunk[-self.overlap_samples:]
            else:
                self.audio_carryover = ready_chunk.copy()
            
            context_prompt = self.token_history[-self.max_tokens:] if self.token_history else None
            
            text, new_tokens = await self.tx._submit_to_queue(
                processing_audio, 
                prefix_tokens=context_prompt, 
                generation_kwargs=self.generation_kwargs
            )
            
            if text:
                self.token_history.extend(new_tokens)
                return {
                    "text": text,
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "is_final_flush": False
                }
        return None

    async def flush_residual_buffer(self) -> Optional[Dict[str, Any]]:
        """Forces immediate final transcription run on lingering bytes right before socket closure."""
        buffer = self.tx.audio_engine.stream_buffer
        if len(buffer) > 0:
            chunk_duration = len(buffer) / self.tx.audio_engine.sample_rate
            audio_array = buffer.numpy()
            
            if self.audio_carryover is not None:
                audio_array = np.concatenate([self.audio_carryover, audio_array])
                
            context_prompt = self.token_history[-self.max_tokens:] if self.token_history else None
            
            text, _ = await self.tx._submit_to_queue(
                audio_array, 
                prefix_tokens=context_prompt, 
                generation_kwargs=self.generation_kwargs
            )
            if text:
                self.flushed_result = {
                    "text": text,
                    "start": round(self.stream_time_offset, 2),
                    "end": round(self.stream_time_offset + chunk_duration, 2),
                    "is_final_flush": True
                }
                return self.flushed_result
        return None




class Transcriber:
    def __init__(
            self,
            whisper_model: AudioTranscriber = None,
            audio_engine: UnifiedAudioEngine = None,
            max_batch_size: int = 8,
            batch_timeout_second: float = 0.05,
            pad_token_id: int = 50257,
            num_token_history: int = 200

    ):
        if whisper_model is None:
            raise AttributeError("WhisperForConditionalGeneration model is not provided!")
        self.model= whisper_model

        if audio_engine is None:
            raise AttributeError("Audio Engine is needed for processing the audio data and mut be provided!")
        self.audio_engine = audio_engine
        self.device = getattr(
            whisper_model,
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_second
        self.pad_token_id = pad_token_id
        self._is_shutting_down = False
        self.num_token_history = num_token_history

        self.worker_task = asyncio.create_task(self._global_batch_processor_worker())


        

    async def _global_batch_processor_worker(self):
        while not self._is_shutting_down or not self.queue.empty():
            try:
                if self._is_shutting_down and self.queue.empty():
                    break


                try:
                    first_request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
            
                requests_batch: List[BatchRequest] = [first_request]
                self.queue.task_done()
                
                start_time = asyncio.get_event_loop().time()
                while len(requests_batch) < self.max_batch_size:
                    time_remaining = self.batch_timeout - (asyncio.get_event_loop().time() - start_time)
                    if time_remaining <= 0:
                        break
                    try:
                        next_request = await asyncio.wait_for(self.queue.get(), timeout=time_remaining)
                        requests_batch.append(next_request)
                        self.queue.task_done()
                    except asyncio.TimeoutError:
                        break
                
                await asyncio.to_thread(self._execute_synchronized_batch, requests_batch)
                
            except asyncio.CancelledError:
                break
            except Exception as worker_error:
                print(f"Error inside global batching worker: {worker_error}")
                await asyncio.sleep(0.1)

    def _execute_synchronized_batch(self, batch: List[BatchRequest]):
        """Executes a single multi-row parallel tensor generation matrix pass on the GPU."""
        all_features = []
        for req in batch:
            features_dict = self.audio_engine.extract_features(req.audio_array)
            feat_tensor = features_dict["input_features"]
            if not isinstance(feat_tensor, torch.Tensor):
                feat_tensor = torch.tensor(feat_tensor)
            all_features.append(feat_tensor)
            
        stacked_features = torch.stack(all_features).to(self.device)
        base_kwargs = batch.generation_kwargs.copy()
        
        has_prefixes = any(req.prefix_tokens for req in batch)
        if has_prefixes:
            max_len = max(len(req.prefix_tokens or []) for req in batch)
            pad_token_id = getattr(
                self.audio_engine.processor.tokenizer,
                "pad_token_id", self.pad_token_id
            )
            
            padded_prefixes = []
            for req in batch:
                tokens = req.prefix_tokens or []
                padded = [pad_token_id] * (max_len - len(tokens)) + tokens
                padded_prefixes.append(padded)
                
            base_kwargs["prompt_ids"] = torch.tensor(padded_prefixes).to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(stacked_features, **base_kwargs)
            
        predicted_list = predicted_ids.tolist()
        
        for idx, req in enumerate(batch):
            row_tokens = predicted_list[idx]
            if has_prefixes:
                prefix_len = len(padded_prefixes[idx])
                if len(row_tokens) > prefix_len:
                    row_tokens = row_tokens[prefix_len:]
                    
            transcription = self.audio_engine.processor.batch_decode(
                [row_tokens], skip_special_tokens=True
            )
            req.result_text = transcription.strip() if transcription else ""
            req.result_tokens = row_tokens
            req.event.set()


    async def _submit_to_queue(
            self,
            audio_array: np.ndarray,
            prefix_tokens: Optional[List[int]],
            generation_kwargs: dict
    ) -> tuple[str, List[int]]:
        """Submits a single tracking payload into the execution queue worker."""
        request = BatchRequest(audio_array, prefix_tokens, generation_kwargs)
        await self.queue.put(request)
        await request.event.wait()
        return request.result_text, request.result_tokens


    def transcribe_file(
            self, audio_input: Any,
            overlap_seconds: float = 0.5,
            **generation_kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Processes offline tracks sequentially with history prompts and acoustic overlap stitching."""
        chunks = self.audio_engine.chunk_offline_file(audio_input)
        history_tokens = []
        
        sample_rate = self.audio_engine.sample_rate
        overlap_samples = int(overlap_seconds * sample_rate)
        leftover_audio_carryover = None
        
        for chunk in chunks:
            truncated_history = history_tokens[-self.num_token_history:] if history_tokens else None
            current_audio = chunk["array"]
            
            if leftover_audio_carryover is not None:
                current_audio = np.concatenate([leftover_audio_carryover, current_audio])
            
            if len(chunk["array"]) >= overlap_samples:
                leftover_audio_carryover = chunk["array"][-overlap_samples:]
            else:
                leftover_audio_carryover = chunk["array"]

            features_dict = self.audio_engine.extract_features(current_audio)
            input_features = features_dict["input_features"]
            if not isinstance(input_features, torch.Tensor):
                input_features = torch.tensor(input_features)
            if input_features.ndim == 2:
                input_features = input_features.unsqueeze(0)
            input_features = input_features.to(self.device)
            
            kwargs = generation_kwargs.copy()
            if truncated_history:
                kwargs["prompt_ids"] = torch.tensor([truncated_history]).to(self.device)
                
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **kwargs)
            
            new_tokens = predicted_ids.tolist()
            if truncated_history and len(new_tokens) > len(truncated_history):
                new_tokens = new_tokens[len(truncated_history):]
                
            transcription = self.audio_engine.processor.batch_decode([new_tokens], skip_special_tokens=True)
            text = transcription.strip() if transcription else ""
            
            if text:
                history_tokens.extend(new_tokens)
                yield {
                    "text": text,
                    "start": round(chunk["start"], 2),
                    "end": round(chunk["end"], 2)
                }

    def stream_session(
            self, 
            silence_timeout_s: float = 1.0, 
            max_context_tokens: int = 150,
            overlap_seconds: float = 0.5,
            **generation_kwargs
    ) -> AsyncStreamContext:
        """Initializes and returns a decoupled, stateful streaming session tracking context."""
        return AsyncStreamContext(
            transcriber=self,
            silence_timeout_s=silence_timeout_s,
            max_context_tokens=max_context_tokens,
            overlap_seconds=overlap_seconds,
            generation_kwargs=generation_kwargs
        )

    # =========================================================================
    # LIFECYCLE CLEANUP METHOD
    # =========================================================================
    async def close(self, timeout_s: float = 5.0):
        """
        Gracefully terminates the background queue processor worker task.
        Drains all remaining queued audio segments before shutting down.
        """
        if self._is_shutting_down:
            return
            
        print("Initiating graceful shutdown of Whisper batch transcriber...")
        self._is_shutting_down = True
        
        try:
            await asyncio.wait_for(self.worker_task, timeout=timeout_s)
            print("Batch transcriber queue drained and closed cleanly.")
        except asyncio.TimeoutError:
            print(f"Shutdown timeout reached after {timeout_s}s. Force-cancelling remaining tasks...")
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass