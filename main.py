# Save inside: main.py

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from utils.transcription_service import TranscriptionService


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "utils")


service = TranscriptionService(config_dir=CONFIG_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.initialize()
    yield
    await service.close(timeout_s=3.0)

app = FastAPI(lifespan=lifespan)

# =========================================================================
# API ROUTE: OFFLINE FILE TRANSCRIPTION (HTTP POST)
# =========================================================================
@app.post("/transcribe/file")
async def transcribe_file(file_path: str, language: str = "en"):
    results = []
    for segment in service.run_offline(file_path, language=language):
        results.append(segment)
    return {"status": "success", "segments": results}

# =========================================================================
# API ROUTE: LIVE REAL-TIME STREAMING (WEBSOCKET)
# =========================================================================
@app.websocket("/transcribe/stream")
async def transcribe_stream(websocket: WebSocket, language: str = "en"):
    await websocket.accept()
    
    with service.create_stream_session(silence_timeout=0.6, language=language) as session:
        try:
            while True:
                frame_bytes = await websocket.receive_bytes()
                result = await session.process_frame(frame_bytes)
                if result:
                    await websocket.send_json(result)
                    
        except WebSocketDisconnect:
            print("WebSocket stream connection terminated cleanly by client.")
        except Exception as ws_error:
            print(f"Active processing anomaly encountered: {ws_error}")
        finally:
            final_chunk = await session.flush_residual_buffer()
            if final_chunk:
                try:
                    await websocket.send_json(final_chunk)
                except Exception:
                    pass  
