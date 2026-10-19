import os
import torch
import asyncio
import numpy as np
from main import service

async def debug_offline_pipeline():
    """Runs a target test on the offline generator pipeline layer."""
    print("\n🔍 [DEBUG] Starting Offline File Transcription Pipeline...")
    
    # Locate a target file from your root layout directory
    target_file = "sample1.flac"
    
    if not os.path.exists(target_file):
        print(f"❌ [DEBUG ERROR] Target track file '{target_file}' not found in directory.")
        return

    try:
        print(f"Reading target metadata file: {target_file}")
        
        # 1. Run the offline execution path generator loop
        # Place a breakpoint on this loop to inspect engine chunk slicing structures!
        generator = service.run_offline(target_file, language="en")
        
        for idx, segment in enumerate(generator):
            print(f"  ├─ Segment #{idx} Captured:")
            print(f"  │  ├─ Time window: {segment['start']}s -> {segment['end']}s")
            print(f"  │  └─ Output Text: '{segment['text']}'")
            
        print("✅ [DEBUG] Offline transcription tracking complete.")
        
    except Exception as e:
        print(f"💥 [DEBUG EXCEPTION] Offline Pipeline Crash: {e}")
        import traceback
        traceback.print_exc()


async def debug_streaming_pipeline():
    """Isolates stream managers to track state data, boundary cuts, and queue jobs."""
    print("\n🔍 [DEBUG] Starting Stateful Real-Time Streaming Pipeline...")
    
    # 1. Generate 3 seconds of silent 16kHz audio data to pass through structural filters
    sample_rate = 16000
    duration_s = 3.0
    total_samples = int(sample_rate * duration_s)
    
    # Create float numpy array wave, then convert to raw 16-bit PCM bytes layout
    mock_wave = np.zeros(total_samples, dtype=np.float32)
    mock_pcm_bytes = (mock_wave * 32768.0).astype(np.int16).tobytes()
    
    # Slice the bytes data block into 100ms fragments mimicking network speed arrivals
    chunk_size_bytes = int(sample_rate * 0.1) * 2  # 100ms * 2 bytes/sample
    bytes_stream_chunks = [
        mock_pcm_bytes[i : i + chunk_size_bytes] 
        for i in range(0, len(mock_pcm_bytes), chunk_size_bytes)
    ]
    
    print(f"Simulating live stream data feed: {len(bytes_stream_chunks)} segments (100ms each)")

    # 2. Build isolated session state wrapper handle
    # Place a breakpoint here to track instance tracking arrays inside AsyncStreamContext!
    with service.create_stream_session(silence_timeout=0.5, language="en") as session:
        try:
            for idx, packet in enumerate(bytes_stream_chunks):
                # Pass single binary frame straight to the state processing hooks
                result = await session.process_frame(packet)
                
                if result:
                    print(f"  🎯 [VAD Pause Boundary Triggered Frame #{idx}]")
                    print(f"     └─ Extracted text block: '{result['text']}'")
                    
                # Short yield pause to let background async orchestration cycles tick
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"💥 [DEBUG EXCEPTION] Active Stream Loop Failure: {e}")
        finally:
            print("  🚨 Stream context loop closing. Forcing final residual wave translation...")
            final_chunk = await session.flush_residual_buffer()
            if final_chunk:
                print(f"  ✨ [Residual Flush Result]: '{final_chunk['text']}'")


async def run_debugger():
    """Entry coordinator running the lifecycle components inside a valid event loop context."""
    print("=========================================================================")
    print("🛠️ WHISPER PIPELINE INFRASTRUCTURE LOGICAL DEBUGGER")
    print("=========================================================================")
    
    # 1. Initialize weights and launch the background queue consumer worker task
    await service.initialize()
    
    # Inspect GPU allocations right after loading weights
    if torch.cuda.is_available():
        print(f"📊 [GPU VRAM Usage]: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated.")
    
    # 2. Run target debugging components sequentially
    await debug_offline_pipeline()
    await debug_streaming_pipeline()
    
    # 3. Terminate background processes gracefully
    print("\n🧹 Cleaning environment dependencies...")
    await service.close(timeout_s=3.0)
    print("=========================================================================")
    print("🏁 Debugging suite run complete.")
    print("=========================================================================")


if __name__ == "__main__":
    # Boots the loop environment context to catch task threads cleanly
    asyncio.run(run_debugger())