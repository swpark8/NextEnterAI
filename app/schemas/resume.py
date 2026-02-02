from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ResumeRequest(BaseModel):
    resume_text: str = Field(..., description="분석할 이력서 원본 텍스트")

class ResumeInputDTO(BaseModel):
    """
    ResumeEngine.recommend() 메소드로 전달되는 통합 입력 객체
    """
    target_role: Optional[str] = None
    resume_content: Dict[str, Any] = {}
    classification: Dict[str, Any] = {}
    evaluation: Dict[str, Any] = {}

    class Config:
        extra = "ignore"

class ResumeAnalysisResponse(BaseModel):
    """
    AI Analyzer가 반환하는 분석 결과 (JSON 구조)
    """
    classification: Dict[str, Any]
    evaluation: Dict[str, Any]

class RecommendationResponse(BaseModel):
    """
    기업 추천 결과 및 AI 리포트
    """
    companies: List[Dict[str, Any]]
    ai_report: Optional[str] = None
