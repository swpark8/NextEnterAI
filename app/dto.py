# Data Transfer Objects
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- 1. [응답] AI -> 클라이언트 (Response DTO) ---

# 1-1. 이력서 분석 결과 (직무 분류)
class ClassificationResult(BaseModel):
    predicted_role: str = Field(..., description="예측된 직무 (예: Backend Developer)")
    keywords: List[str] = Field(default_factory=list, description="이력서 핵심 키워드")

# 1-2. 이력서 분석 결과 (심층 평가)
class EvaluationResult(BaseModel):
    grade: str = Field(..., description="등급 (S/A/B/C)")
    score: int = Field(..., description="점수 (0-100)")
    summary: str = Field(..., description="평가 요약 (한줄평)")
    pros: List[str] = Field(default_factory=list, description="장점 (강점)")
    cons: List[str] = Field(default_factory=list, description="단점 (보완점)")
    recommended_companies: List[str] = Field(default_factory=list, description="단순 텍스트 추천 기업")

# 1-3. [분석 API] 최종 응답
class ResumeAnalysisResponse(BaseModel):
    classification: ClassificationResult
    evaluation: EvaluationResult

# 1-4. [추천 API] 기업 정보 구조
class CompanyInfo(BaseModel):
    company_name: str
    role: str
    score: float
    is_exact_match: bool

# 1-5. [추천 API] 최종 응답
class RecommendationResponse(BaseModel):
    companies: List[CompanyInfo]
    ai_report: str

# --- 2. [요청] 클라이언트 -> AI (Request DTO) ---

# 2-1. 분석 요청 (텍스트만 줌)
class ResumeRequest(BaseModel):
    resume_text: str

# 2-2. 추천 요청 (JSON 구조체 줌)
class ResumeInputDTO(BaseModel):
    id: str
    target_role: str
    resume_content: Dict[str, Any] 
    # 예: { "skills": ["Java"], "experience": [...] }