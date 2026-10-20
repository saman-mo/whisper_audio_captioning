import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status, Request, Depends
from schemas import (
    TranscribeFileRequest, 
    TranscribeFileResponse, 
    StreamSessionInit, 
    LiveStreamSegment
)
from utils.transcription_service import TranscriptionService


router = APIRouter(prefix="/api/v1/transcribe", tags=["Transcription"])

def get_transcription_service(request: Request) -> TranscriptionService:
    """Dependency injection to provide type tracking for VS Code / Pylance."""
    return request.app.state.transcription_service

@router.post("/file", response_model=TranscribeFileResponse, status_code=status.HTTP_200_OK)
async def transcribe_file(
    payload: TranscribeFileRequest, 
    service: TranscriptionService = Depends(get_transcription_service)
):
    """Processes an offline local audio file using background transcription layers."""
    try:
        results = []
        for segment in service.run_offline(payload.file_path, language=payload.language):
            results.append(segment)
        return TranscribeFileResponse(status="success", segments=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process offline file: {str(e)}"
        )


@router.websocket("/stream")
async def transcribe_stream(websocket: WebSocket, language: str = "en"):
    """Accepts real-time streaming raw audio chunks via binary WebSocket frames."""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    # Retrieve the state manually inside WebSocket since Depends is built for HTTP verbs
    service: TranscriptionService = websocket.app.state.transcription_service
    
    # Notify user node of assigned session tracking reference metrics via our contract model
    init_payload = StreamSessionInit(session_id=session_id)
    await websocket.send_json(init_payload.model_dump())
    
    with service.create_stream_session(silence_timeout=0.6, language=language) as session:
        try:
            while True:
                frame_bytes = await websocket.receive_bytes()
                result = await session.process_frame(frame_bytes)
                if result:
                    # Explicitly validate the streaming payload chunk dynamically
                    validated_segment = LiveStreamSegment(
                        start=result["start"],
                        end=result["end"],
                        text=result["text"],
                        is_final_flush=result.get("is_final_flush", False)
                    )
                    await websocket.send_json(validated_segment.model_dump())
                    
                    # Automatically stream-append into distributed Redis memory cache
                    await service.save_segment(session_id=session_id, segment=result)
                    
        except WebSocketDisconnect:
            print(f"WebSocket session {session_id} terminated cleanly by client.")
        except Exception as ws_error:
            print(f"Active processing anomaly encountered on {session_id}: {ws_error}")
        finally:
            final_chunk = await session.flush_residual_buffer()
            if final_chunk:
                try:
                    validated_final = LiveStreamSegment(
                        start=final_chunk["start"],
                        end=final_chunk["end"],
                        text=final_chunk["text"],
                        is_final_flush=True
                    )
                    await websocket.send_json(validated_final.model_dump())
                    await service.save_segment(session_id=session_id, segment=final_chunk)
                except Exception:
                    pass