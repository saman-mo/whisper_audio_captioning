from fastapi import APIRouter, HTTPException, Query, Request, status, Depends

from schemas import SearchResponse, SummarizeRequest, SummarizeResponse
from utils import TranscriptionService
from utils import InferenceEngineException


router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

def get_transcription_service(request: Request) -> TranscriptionService:
    """Dependency injection to provide type tracking for VS Code / Pylance."""
    return request.app.state.transcription_service




@router.get("/{session_id}/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_session_transcript(
    session_id: str,
    q: str = Query(..., min_length=1, description="The keyphrase query to search for"),
    service: TranscriptionService = Depends(get_transcription_service)
):
    """
    Unified search cluster endpoint. Scans conversation transcript segments 
    by session ID across any distributed cluster node for both real-time streams 
    and processed offline files.
    """
    try:
        matches = await service.search_transcript(session_id=session_id, query=q)
        
        return SearchResponse(
            session_id=session_id,
            query=q,
            total_matches=len(matches),
            matches=matches
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Session {session_id} not found, has expired, or contains no transcripts yet."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal execution exception thrown during cluster cache search lookup."
        )



@router.post("/{session_id}/summarize", response_model=SummarizeResponse, status_code=status.HTTP_200_OK)
async def summarize_session_transcript(
    session_id: str,
    payload: SummarizeRequest,
    service: TranscriptionService = Depends(get_transcription_service)
):
    """
    Unified abstractive text summarization endpoint. Chronologically extracts 
    historical text segments from the session cache and prompts the downstream 
    NLP engine to generate structured summaries for live streams or offline files.
    """
    try:
        summary_text = await service.summarize_transcript(session_id=session_id, format_type=payload.format.value)
        return SummarizeResponse(session_id=session_id, summary=summary_text)
        
    except KeyError:
        # Catch missing, expired, or uninitialized cache sessions
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Session text arrays for {session_id} not found or expired."
        )
    except InferenceEngineException as inference_err:
        # INTERCEPT CUSTOM INFERENCE FAULT: Map it to an explicit HTTP 502 Bad Gateway response layout
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream LLM Cluster Failure: {str(inference_err)}"
        )
    except Exception as general_err:
        # Catch true application failures or unhandled exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Core inference processing execution exception: {str(general_err)}"
        )
        