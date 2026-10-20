from fastapi import APIRouter, HTTPException, Query, Request, status, Depends
from schemas import SearchResponse, SummarizeRequest, SummarizeResponse
from utils import TranscriptionService

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
    Distributed search cluster endpoint. Scans a live conversation transcript 
    by session across any instance node via distributed memory maps.
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
    try:
        summary_text = await service.summarize_transcript(session_id=session_id, format_type=payload.format.value)
        return SummarizeResponse(session_id=session_id, summary=summary_text)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session text arrays for {session_id} not found or expired.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Core inference processing execution exception: {str(e)}")