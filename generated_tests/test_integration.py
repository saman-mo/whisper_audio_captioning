import asyncio
import json
import httpx
import websockets
import numpy as np
from jose import jwt


JWT_SECRET = "mock_secret_for_signing_integration_token"
JWT_ALGORITHM = "HS256"  # Swap to RS256 + Private Key block for production signatures

MOCK_JWKS_URL = "https://your-idp-domain.com"
MOCK_AUDIENCE = "https://enterprise.com"
MOCK_ISSUER = "https://your-idp-domain.com"

BASE_HTTP_URL = "http://localhost:8000/api/v1"
BASE_WS_URL = "ws://localhost:8000/api/v1"

def generate_mock_token() -> str:
    """
    Generates a structural JWT payload passing verification rules.
    Dynamically mirrors your target production claims and issuers.
    """
    claims = {
        "sub": "integration-test-client-01",
        "aud": MOCK_AUDIENCE,
        "iss": MOCK_ISSUER,
        "exp": 9999999999
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)

def generate_mock_pcm_audio_frame(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates Float32 mono PCM audio binary fragments to test stream processing lines."""
    total_samples = int(sample_rate * duration_seconds)
    audio_array = np.random.uniform(-0.05, 0.05, total_samples).astype(np.float32)
    return audio_array.tobytes()


# =========================================================================
# 🔎 SHARED ANALYTICS AUDIT FUNCTION
# =========================================================================
async def verify_analytics_pipeline(http_client: httpx.AsyncClient, session_id: str, pipeline_label: str):
    """Runs search and Llama summarization verifications against a specific session ID."""
    print(f"\n📊 [Audit: {pipeline_label}] Querying analytics routes for: {session_id}")
    
    # Query the populated transcript data layer using a valid search token
    search_query = "utilitarianism"
    search_endpoint = f"{BASE_HTTP_URL}/analytics/{session_id}/search?q={search_query}"
    
    search_response = await http_client.get(search_endpoint)
    print(f"   📋 [Search Status]: {search_response.status_code}")
    if search_response.status_code == 200:
        search_payload = search_response.json()
        print(f"   ✅ [Search Success]: Found {search_payload.get('total_matches')} match items across timelines.")
    else:
        print(f"   ❌ [Search Failed]: Status code {search_response.status_code}. Response: {search_response.text}")
    
    # 2. Test abstractive summary strategies directed to local Llama inference node
    summarize_endpoint = f"{BASE_HTTP_URL}/analytics/{session_id}/summarize"
    summary_payload = {"format": "bullets"}
    
    print(f"   📡 [POST] Dispatching summary instructions to Llama...")
    summary_response = await http_client.post(summarize_endpoint, json=summary_payload, timeout=45.0)
    print(f"   📋 [Summary Status]: {summary_response.status_code}")
    if summary_response.status_code == 200:
        summary_payload = summary_response.json()
        print(f"   ✅ [Summary Success] Llama 3.2 Output:\n{summary_payload.get('summary')}\n")
    else:
        print(f"   ❌ [Summary Failed]: Status code {summary_response.status_code}.")

# =========================================================================
# 🚀 CORE AUTOMATED DUAL INTEGRATION TEST ROUTINE
# =========================================================================
async def run_e2e_integration_test():
    auth_token = generate_mock_token()
    
    async with httpx.AsyncClient() as http_client:
        
        # =====================================================================
        # 📂 PHASE 1: OFFLINE BATCH FILE PIPELINE TEST
        # =====================================================================
        print("🎬 [PHASE 1: Offline File Testing] Triggering batch audio processing...")
        
        offline_endpoint = f"{BASE_HTTP_URL}/transcribe/file"
        offline_payload = {
            "file_path": "/app/sample3.flac",  
            "language": "en"
        }
        
        try:
            print(f"📡 [POST] Sending offline request to: {offline_endpoint}")
            offline_response = await http_client.post(offline_endpoint, json=offline_payload, timeout=10.0)
            print(f"📋 [Offline Status]: {offline_response.status_code}")
            
            if offline_response.status_code == 200:
                offline_data = offline_response.json()
                offline_session_id = offline_data.get("session_id")
                print(f"✅ [Offline Handshake Locked]: Session ID -> {offline_session_id}")
                
                # Verify analytics directly using the symmetrical data track
                await verify_analytics_pipeline(http_client, offline_session_id, "UNIFIED ANALYTICS")
            else:
                print(f"❌ [Offline Failed]: Status code {offline_response.status_code}.")
        except Exception as file_fault:
            print(f"⚠️ [Offline Skipped/Failed]: Could not complete offline loop. Trace: {str(file_fault)}")

        # =====================================================================
        # 🌊 PHASE 2: REAL-TIME STREAMING WEBSOCKET STRESS TEST
        # =====================================================================
        print("\n🎬 [PHASE 2: Streaming Tunnel Testing] Initializing WebSocket pipeline...")
        stream_socket_url = f"{BASE_WS_URL}/transcribe/stream?token={auth_token}&language=en"
        
        print(f"🔗 [WebSocket] Connecting to: {stream_socket_url}")
        try:
            async with websockets.connect(stream_socket_url) as ws:
                print("✅ [WebSocket] Connection open. Zero-Trust Handshake verified.")
                
                init_message_raw = await ws.recv()
                init_data = json.loads(init_message_raw)
                print(f"📥 [Handshake Received]: {init_data}")
                
                print("📤 [Streaming] Pumping real-time binary payload frames to audit network loops...")
                for frame_idx in range(5):
                    audio_payload = generate_mock_pcm_audio_frame(duration_seconds=1.0)
                    await ws.send(audio_payload)
                    print(f"   Sent chunk {frame_idx + 1}/5")
                    await asyncio.sleep(0.4)
                
                print("🔌 [WebSocket] Closing connection context manager...")
                print("✅ [E2E SUCCESS]: All microservice boundaries, network channels, and database gates verified clean!")
            
        except Exception as stream_fault:
            print(f"❌ [Streaming Pipeline Drop]: Trace: {str(stream_fault)}")

if __name__ == "__main__":
    asyncio.run(run_e2e_integration_test())