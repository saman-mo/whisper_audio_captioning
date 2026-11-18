import os
import sys
import asyncio
import json
import httpx
import websockets
import numpy as np
import soundfile as sf
from jose import jwt


# KUBERNETES CLUSTER & AUTH CONFIGURATION

# Pass minikube tunnel URL via CLI: python test_k8s_mesh.py http://127.0.0.1:54321
# Or leave default to hit NodePort 30080 directly (e.g., via minikube ip / port-forward)
RAW_TARGET = sys.argv[1] if len(sys.argv) > 1 else os.getenv("K8S_SERVICE_URL", "http://127.0.0.1:30080")
CLEAN_HOST = RAW_TARGET.replace("http://", "").replace("https://", "").rstrip("/")

BASE_ROOT_URL = f"http://{CLEAN_HOST}"
BASE_HTTP_URL = f"http://{CLEAN_HOST}/api/v1"
BASE_WS_URL = f"ws://{CLEAN_HOST}/api/v1"

JWT_SECRET = "mock_secret_for_signing_integration_token"
JWT_ALGORITHM = "HS256"
MOCK_AUDIENCE = "https://enterprise.com"
MOCK_ISSUER = "https://your-idp-domain.com"

LOCAL_HOST_AUDIO_PATH = "test_audio.wav"
CONTAINER_OFFLINE_AUDIO = "/app/sample3.flac"


def generate_mock_token() -> str:
    claims = {
        "sub": "k8s-integration-client-01",
        "aud": MOCK_AUDIENCE,
        "iss": MOCK_ISSUER,
        "exp": 9999999999
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)


def generate_mock_int16_frame(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates Int16 mono PCM bytes matching UnifiedAudioEngine.load_and_resample."""
    total_samples = int(sample_rate * duration_seconds)
    audio_array = np.random.randint(-1500, 1500, size=total_samples, dtype=np.int16)
    return audio_array.tobytes()



# STEP 0: CLUSTER HEALTH & POD READINESS CHECK

async def verify_cluster_health(http_client: httpx.AsyncClient) -> bool:
    print(f"🩺 [STEP 0: Cluster Probe] Checking /health on {BASE_ROOT_URL}...")
    try:
        resp = await http_client.get(f"{BASE_ROOT_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            print("   ✅ [Healthy] FastAPI backend pod responded 200 OK.\n")
            return True
        print(f"   ❌ [Unhealthy] Unexpected status code: {resp.status_code}")
        return False
    except Exception as e:
        print(f"   ❌ [Connection Refused] Could not reach {BASE_ROOT_URL}. Is `minikube service` or `kubectl port-forward` running?\n   Trace: {e}")
        return False



# SHARED REDIS & LLAMA ANALYTICS VERIFICATION

async def verify_analytics_pipeline(http_client: httpx.AsyncClient, session_id: str, label: str, search_query: str = "utilitarianism"):
    print(f"\n📊 [Analytics Audit: {label}] Session ID: {session_id}")
    
    # 1. Redis Transcript Search
    search_url = f"{BASE_HTTP_URL}/analytics/{session_id}/search?q={search_query}"
    search_resp = await http_client.get(search_url, timeout=15.0)
    if search_resp.status_code == 200:
        data = search_resp.json()
        print(f"   ✅ [Redis Search] Status 200 | Matches for '{search_query}': {data.get('total_matches', 0)}")
    else:
        print(f"   ❌ [Redis Search] Failed ({search_resp.status_code}): {search_resp.text}")

    # 2. Local Llama 3.2 Summarization (Extended timeout for CPU inference)
    summarize_url = f"{BASE_HTTP_URL}/analytics/{session_id}/summarize"
    print("   ⏳ [Llama 3.2] Dispatching summary request to local-llama-server (CPU inference may take 15-60s)...")
    sum_resp = await http_client.post(summarize_url, json={"format": "bullets"}, timeout=120.0)
    if sum_resp.status_code == 200:
        summary = sum_resp.json().get("summary", "")
        print(f"   ✅ [Llama Summary] Success:\n{summary}\n")
    else:
        print(f"   ❌ [Llama Summary] Failed ({sum_resp.status_code}): {sum_resp.text}\n")



# STEP 1: OFFLINE FILE PIPELINE TEST

async def test_offline_pipeline(http_client: httpx.AsyncClient):
    print("🎬 [STEP 1: Offline Batch Endpoint] Testing /transcribe/file...")
    url = f"{BASE_HTTP_URL}/transcribe/file"
    payload = {"file_path": CONTAINER_OFFLINE_AUDIO, "language": "en"}
    
    try:
        resp = await http_client.post(url, json=payload, timeout=120.0)
        if resp.status_code == 200:
            session_id = resp.json().get("session_id")
            print(f"   ✅ [Offline Transcribed] Session ID: {session_id}")
            await verify_analytics_pipeline(http_client, session_id, "OFFLINE BATCH")
        else:
            print(f"   ⚠️ [Offline Skipped] Endpoint returned {resp.status_code} (Verify '{CONTAINER_OFFLINE_AUDIO}' exists inside the pod): {resp.text}\n")
    except Exception as e:
        print(f"   ⚠️ [Offline Error] {e}\n")



# STEP 2: REAL-TIME WEBSOCKET STREAMING TEST

async def test_streaming_pipeline(http_client: httpx.AsyncClient, token: str):
    print("🎬 [STEP 2: Real-Time WebSocket Stream] Testing /transcribe/stream...")
    ws_url = f"{BASE_WS_URL}/transcribe/stream?token={token}&language=en"
    
    use_real_audio = os.path.exists(LOCAL_HOST_AUDIO_PATH)
    if use_real_audio:
        print(f"   🎙️ Loaded local audio file: '{LOCAL_HOST_AUDIO_PATH}'")
        audio_data, _ = sf.read(LOCAL_HOST_AUDIO_PATH, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_int16 = (audio_data.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        print(f"   ⚠️ '{LOCAL_HOST_AUDIO_PATH}' not found. Falling back to 5s synthetic Int16 PCM noise frames.")

    session_id = None
    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=60) as ws:
            init_data = json.loads(await ws.recv())
            session_id = init_data.get("session_id")
            print(f"   ✅ [WS Handshake] Connected | Session ID: {session_id}")

            if use_real_audio:
                samples_per_sec = 16000
                for idx, start in enumerate(range(0, len(audio_int16), samples_per_sec), start=1):
                    chunk = audio_int16[start:start + samples_per_sec].tobytes()
                    await ws.send(chunk)
                    print(f"   📤 Sent real audio chunk {idx}")
                    await asyncio.sleep(0.5)
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.05))
                        print(f"      📥 [Live Transcript]: {msg.get('text')}")
                    except asyncio.TimeoutError:
                        pass
            else:
                for idx in range(1, 6):
                    await ws.send(generate_mock_int16_frame(1.0))
                    print(f"   📤 Sent synthetic PCM chunk {idx}/5")
                    await asyncio.sleep(0.4)

            print("   🔌 Closing WebSocket connection (triggering residual buffer flush)...")

    except Exception as e:
        print(f"   ❌ [WebSocket Error] {e}")
        return

    if session_id and use_real_audio:
        print("   ⏳ Waiting 6s for CPU batch worker to finish final flush & Redis write...")
        await asyncio.sleep(6.0)
        await verify_analytics_pipeline(http_client, session_id, "LIVE STREAM")
    elif session_id:
        print("   ℹ️ Synthetic noise contained no speech; skipping Llama summary check for stream session.\n")


# =========================================================================
# ENTRYPOINT
# =========================================================================
async def main():
    token = generate_mock_token()
    async with httpx.AsyncClient() as http_client:
        if not await verify_cluster_health(http_client):
            return
        await test_offline_pipeline(http_client)
        await test_streaming_pipeline(http_client, token)
        print("🎉 [ALL KUBERNETES MESH CHECKS COMPLETE]")


if __name__ == "__main__":
    asyncio.run(main())
