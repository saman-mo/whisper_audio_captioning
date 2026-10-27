import uuid
import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status, Query, Depends, Request
from jose import jwt, JWTError
from schemas import (
    TranscribeFileRequest, 
    TranscribeFileResponse, 
    StreamSessionInit, 
    LiveStreamSegment
)
from utils.transcription_service import TranscriptionService


router = APIRouter(prefix="/api/v1/transcribe", tags=["Transcription"])

JWKS_CACHE = {}


def get_transcription_service_http(request: Request) -> TranscriptionService:
    """Dependency injection to provide type tracking for HTTP endpoints (REST)."""
    return request.app.state.transcription_service

def get_transcription_service_ws(websocket: WebSocket) -> TranscriptionService:
    """Dependency injection to provide type tracking for WebSocket endpoints."""
    return websocket.app.state.transcription_service


async def verify_jwt_token(token: str, jwks_url: str, audience: str, issuer: str, enforce: bool) -> bool:
    """
    Cryptographically verifies an asymmetric RS256 JWT signature using 
    dynamic public keys retrieved from an enterprise Identity Provider (IdP).
    """
    if not enforce:
        return True
        
    global JWKS_CACHE
    try:
        # 1. Unparse the token header to check which key ID ('kid') signed it
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            print("JWT Validation Failure: Missing 'kid' header field.")
            return False

        # 2. Fetch public keys if not cached locally
        if not JWKS_CACHE or kid not in JWKS_CACHE:
            async with httpx.AsyncClient() as client:
                response = await client.get(jwks_url, timeout=5.0)
                response.raise_for_status()
                jwks = response.json()
                JWKS_CACHE = {key["kid"]: key for key in jwks.get("keys", [])}

        public_key = JWKS_CACHE.get(kid)
        if not public_key:
            print("JWT Validation Failure: No matching public key found in cluster JWKS.")
            return False

        # 3. Perform cryptographic token decryption signature validation
        jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer
        )
        return True

    except JWTError as crypto_exception:
        print(f"Cryptographic validation rejection signature: {str(crypto_exception)}")
        return False
    except Exception as network_exception:
        print(f"Identity Provider unreachable or structurally invalid: {str(network_exception)}")
        return False



# 1. OFFLINE FILE TRANSLATION ENDPOINT (HTTP REST)
@router.post("/file", response_model=TranscribeFileResponse, status_code=status.HTTP_200_OK)
async def transcribe_file(
    payload: TranscribeFileRequest, 
    service: TranscriptionService = Depends(get_transcription_service_http)
):
    """
    Processes an offline local audio file, pushes the chronological 
    segments into the atomic Redis cache, and exposes search capabilities.
    """
    try:
        results = []
        # 1. Generate a valid session tracking key for this file run
        session_id = f"offline-{uuid.uuid4()}"
        
        for segment in service.run_offline(payload.file_path, language=payload.language):
            results.append(segment)
            
            # 2. Push segments directly into an atomic Redis list layer
            await service.save_segment(session_id=session_id, segment=segment)
            
        # 3. Return the session_id back to the client app so they can run searches
        return TranscribeFileResponse(
            status="success", 
            session_id=session_id,
            segments=results
        )
    except MemoryError as memory_fault:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=f"Database storage capacity limit exceeded: {str(memory_fault)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process offline file: {str(e)}"
        )


# 2. REAL-TIME STREAMING ENDPOINT (WEBSOCKETS)
# =========================================================================
@router.websocket("/stream")
async def transcribe_stream(
    websocket: WebSocket,
    token: str = Query(..., description="Secure bearer JWT access token"),
    language: str = "en",
    service: TranscriptionService = Depends(get_transcription_service_ws)
):
    """
    Hardened real-time streaming endpoint driven by Zero-Trust JWT verification rules.
    Connections snap shut instantly if crypto validation metrics fail to pass.
    """
    security_cfg = service.config.security
    max_chunk_size = security_cfg.max_chunk_size_bytes
    enforce_auth = security_cfg.enforce_token_auth
    jwks_url = security_cfg.jwks_url
    audience = security_cfg.audience
    issuer = security_cfg.issuer

    # 1. Cryptographic Authentication Handshake Validation Check
    is_valid = await verify_jwt_token(token, jwks_url, audience, issuer, enforce_auth)
    if not is_valid:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        print("WebSocket rejected: Cryptographic signature validation failure.")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"[Authenticated via IdP JWT] Initialized secure session workspace: {session_id}")
    
    init_payload = StreamSessionInit(session_id=session_id)
    await websocket.send_json(init_payload.model_dump())
    
    try:
        with service.create_stream_session(silence_timeout=0.6, language=language) as session:
            while True:
                frame_bytes = await websocket.receive_bytes()
                
                # 2. Parameterized Defensive Clamping
                if len(frame_bytes) > max_chunk_size:
                    print(f"Session {session_id} frame size ({len(frame_bytes)} B) exceeded threshold ({max_chunk_size} B).")
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    break
                
                result = await session.process_frame(frame_bytes)
                if result:
                    validated_segment = LiveStreamSegment(
                        start=result["start"],
                        end=result["end"],
                        text=result["text"],
                        is_final_flush=result.get("is_final_flush", False)
                    )
                    await websocket.send_json(validated_segment.model_dump())
                    try:
                        await service.save_segment(session_id=session_id, segment=result)
                    except MemoryError:
                        print(f"Terminating stream session {session_id} due to host OOM.")
                        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                        break
                    
    except WebSocketDisconnect:
        print(f"Clean disconnection processed for session: {session_id}")
    except Exception as stream_fault:
        print(f"Critical exception thrown inside stream channel [{session_id}]: {str(stream_fault)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        print(f"Commencing stream session drain down protocols for: {session_id}")
        try:
            final_chunk = await session.flush_residual_buffer()
            if final_chunk:
                validated_final = LiveStreamSegment(
                    start=final_chunk["start"],
                    end=final_chunk["end"],
                    text=final_chunk["text"],
                    is_final_flush=True
                )
                await websocket.send_json(validated_final.model_dump())
                await service.save_segment(session_id=session_id, segment=final_chunk)
        except Exception as drain_exception:
            print(f"Non-blocking residual flush failure encountered: {drain_exception}")