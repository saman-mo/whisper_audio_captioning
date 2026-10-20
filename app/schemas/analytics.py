from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class SearchMatch(BaseModel):
    segment_index: int = Field(..., description="Index of the segment within the transcript")
    start_time: float = Field(..., description="Start timestamp of the matching audio segment in seconds")
    end_time: float = Field(..., description="End timestamp of the matching audio segment in seconds")
    text: str = Field(..., description="The full raw text of the segment")
    match_offsets: List[int] = Field(..., description="Character indices where the search query starts inside the text")

class SearchResponse(BaseModel):
    session_id: str = Field(..., description="The unique session identifier evaluated")
    query: str = Field(..., description="The original string query passed to the search engine")
    total_matches: int = Field(..., description="Total count of matching text occurrences")
    matches: List[SearchMatch] = Field(..., description="Detailed timeline matrix of matches")

class SummaryFormat(str, Enum):
    BULLETS = "bullets"
    PARAGRAPH = "paragraph"
    ACTION_ITEMS = "action_items"

class SummarizeRequest(BaseModel):
    format: SummaryFormat = Field(SummaryFormat.BULLETS, description="The desired presentation structure of the text summary")

class SummarizeResponse(BaseModel):
    session_id: str = Field(..., description="The unique session identifier evaluated")
    summary: str = Field(..., description="The fully generated abstractive text summary outcome")