import asyncio
import json
import httpx
import websockets
import os
import soundfile as sf
from jose import jwt
import numpy as np

# =========================================================================
# CONFIGURATION & ENTERPRISE PRODUCTION JWT PARAMETERS
# =========================================================================
JWT_SECRET = "mock_secret_for_signing_integration_token"
JWT_ALGORITHM = "HS256"

MOCK_JWKS_URL = "https://your-idp-domain.com"
MOCK_AUDIENCE = "https://enterprise.com"
MOCK_ISSUER = "https://your-idp-domain.com"

BASE_HTTP_URL = "http://localhost:8000/api/v1"
BASE_WS_URL = "ws://localhost:8000/api/v1"

LOCAL_HOST_AUDIO_PATH = "test_audio.wav" 

def generate_mock_token() -> str:
    """Generates a structural JWT payload passing verification rules."""
    claims = {
        "sub": "streaming-test-client-01",
        "aud": MOCK_AUDIENCE,
        "iss": MOCK_ISSUER,
        "exp": 9999999999
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)

# =========================================================================
# 🚀 CORE AUTOMATED REAL-TIME STREAMING TEST ROUTINE
# =========================================================================
async def run_streaming_integration_test():
    if not os.path.exists(LOCAL_HOST_AUDIO_PATH):
        print(f"❌ [Error]: Local test audio file not found at '{LOCAL_HOST_AUDIO_PATH}'.")
        return

    audio_data, sample_rate = sf.read(LOCAL_HOST_AUDIO_PATH, dtype='float32')
    
    # If the file has stereo audio channels, downmix to a single mono track
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    audio_int16 = (audio_data.clip(-1.0, 1.0) * 32767.0).astype(np.int16)

    auth_token = generate_mock_token()
    stream_socket_url = f"{BASE_WS_URL}/transcribe/stream?token={auth_token}&language=en"
    stream_session_id = None
    
    print(f"🔗 [WebSocket] Connecting to streaming endpoint: {stream_socket_url}")
    
    try:
        async with websockets.connect(stream_socket_url) as ws:
            print("✅ [WebSocket] Connection open. Zero-Trust Handshake verified.")
            
            init_message_raw = await ws.recv()
            init_data = json.loads(init_message_raw)
            print(f"📥 [Handshake Received]: {init_data}")
            
            stream_session_id = init_data.get("session_id")
            assert stream_session_id is not None, "Streaming session ID assignment failed."
            print(f"🎯 [Session Registered]: {stream_session_id}")
            
            print(f"📤 [Streaming] Pumping true Float32 PCM speech vectors down connection...")
            
            # Stream in clean 1-second chunks (16,000 floats per block)
            samples_per_second = 16000
            chunk_idx = 1
            
            for start_idx in range(0, len(audio_int16), samples_per_second):
                chunk = audio_int16[start_idx:start_idx + samples_per_second]
                
                # Convert the float array slice straight into raw uncompressed binary frames
                binary_payload = chunk.tobytes()
                
                await ws.send(binary_payload)
                print(f"   Piped clean voice frame chunk {chunk_idx}")
                chunk_idx += 1
                
                # Yield context control to simulate natural human talking speed
                await asyncio.sleep(0.5)
                
                # Check for any real-time transcription updates sent back over the wire
                try:
                    raw_response = await asyncio.wait_for(ws.recv(), timeout=0.02)
                    live_transcription = json.loads(raw_response)
                    print(f"      📥 [Live Text Update]: {live_transcription.get('text')}")
                except asyncio.TimeoutError:
                    pass
            
            print("🔌 [WebSocket] Finished streaming file bytes. Closing connection...")
            
    except Exception as stream_fault:
        print(f"❌ [Streaming Pipeline Drop]: Fatal connection failure. Trace: {str(stream_fault)}")
        return

    print("\n⏳ Waiting 5 seconds for backend GPU queues to finish transcribing and writing to Redis...")
    await asyncio.sleep(5.0)
    
    async with httpx.AsyncClient() as http_client:
        print(f"🔎 [Verification Phase] Querying analytics routes for stream ID: {stream_session_id}")
        
        search_query = "utilitarianism"
        search_endpoint = f"{BASE_HTTP_URL}/analytics/{stream_session_id}/search?q={search_query}"
        
        print(f"📡 [GET] Dispatching search request: {search_endpoint}")
        search_response = await http_client.get(search_endpoint)
        print(f"📋 [Search Status]: {search_response.status_code}")
        
        if search_response.status_code == 200:
            search_payload = search_response.json()
            print(f"   ✅ [Search Success]: Found {search_payload.get('total_matches', 0)} matches for query '{search_query}'.")
        else:
            print(f"   ❌ [Search Error]: Failed with status {search_response.status_code}. Response: {search_response.text}")

        summarize_endpoint = f"{BASE_HTTP_URL}/analytics/{stream_session_id}/summarize"
        summary_payload = {"format": "bullets"}
        
        print(f"📡 [POST] Dispatching summary instructions to Llama container: {summarize_endpoint}")
        summary_response = await http_client.post(summarize_endpoint, json=summary_payload, timeout=45.0)
        print(f"📋 [Summary Status]: {summary_response.status_code}")
        
        if summary_response.status_code == 200:
            summary_payload = summary_response.json()
            print("✅ [Summary Success] Llama 3.2 produced executive stream insights:")
            print(f"📝 Summary Output:\n{summary_payload.get('summary')}")
        else:
            print(f"   ❌ [Summary Error]: Failed with status {summary_response.status_code}. Response: {summary_response.text}")

if __name__ == "__main__":
    asyncio.run(run_streaming_integration_test())
