from pydantic import BaseModel, Field
from typing import List

class TranscribeFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to local target audio file on storage volumes")
    language: str = Field("en", description="ISO 639-1 language translation configuration string")

class SpeechSegment(BaseModel):
    start: float = Field(..., description="Start window of speech activity chunk in seconds")
    end: float = Field(..., description="End window of speech activity chunk in seconds")
    text: str = Field(..., description="The raw transcribed alpha string segment mapping text contents")

class TranscribeFileResponse(BaseModel):
    status: str = Field("success", description="Functional operation confirmation flag")
    segments: List[SpeechSegment] = Field(..., description="Chronological array elements indexing segments")

class LiveStreamSegment(BaseModel):
    start: float = Field(..., description="Start timestamp of the real-time chunk")
    end: float = Field(..., description="End timestamp of the real-time chunk")
    text: str = Field(..., description="The raw real-time transcribed text string")
    is_final_flush: bool = Field(..., description="Flags if this is the absolute final chunk of a buffer stream")

class StreamSessionInit(BaseModel):
    type: str = Field("session_init", description="Internal handshake step identifier")
    session_id: str = Field(..., description="The uniquely generated cluster session string for tracking")