from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. Domain/Internal Models (Originally from schema.py)
# ==========================================

class Education(BaseModel):
    degree: str
    major: str
    status: str

class Skills(BaseModel):
    essential: List[str]
    additional: List[str]

class Experience(BaseModel):
    company: str
    period: str
    role: str
    key_tasks: List[str]

class ProjectExperience(BaseModel):
    project_title: str
    key_achievements: List[str]

class ResumeContent(BaseModel):
    education: List[Education]
    skills: Skills
    professional_experience: List[Experience]
    project_experience: Optional[List[ProjectExperience]] = []

class StandardizedRole(BaseModel):
    category: str
    title: str
    level: str

# 평가 데이터 내부 구조
class AtsScore(BaseModel):
    S_matched: int
    S_required: int
    final_ats: float
    explanation: Optional[str] = None

class ResumeEvaluation(BaseModel):
    job_title: str
    grade: str
    score: float
    ats_score: Optional[AtsScore] = None
    reasoning: Optional[str] = None

# 메인 이력서 모델 (Resume)
class Resume(BaseModel):
    id: str
    candidate_id: Optional[str] = None
    target_role: str
    standardized_role: StandardizedRole
    resume_content: ResumeContent
    resume_evaluation: Optional[ResumeEvaluation] = None

# 회사(Company) 모델
class Company(BaseModel):
    name: str
    tier: str
    employee_count: Optional[str] = None
    industry: str
    location: str
    tech_stack: List[str]
    target_roles: List[str]

# API 응답 및 메타데이터 모델 (from schema.py)
class MatchResult(BaseModel):
    company_name: str
    match_score: float
    tier: str
    match_type: str
    reason: str

# [추가] final_metadata_600.json 파일 구조 검증용
class ResumeMetadata(BaseModel):
    resume_id: str
    target_role: str
    recommended_companies: List[MatchResult]


# ==========================================
# 2. API DTO Models (Originally from dto.py)
# ==========================================

# --- [응답] AI -> 클라이언트 (Response DTO) ---

# 1-1. 이력서 분석 결과 (직무 분류)
class ClassificationResult(BaseModel):
    predicted_role: str = Field(..., description="예측된 직무 (예: Backend Developer)")
    keywords: List[str] = Field(default_factory=list, description="이력서 핵심 키워드")

# 1-2. 이력서 분석 결과 (심층 평가)
class EvaluationResult(BaseModel):
    grade: str = Field(..., description="등급 (S/A/B/C/F)")
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
    match_level: str  # BEST, HIGH, LOW
    is_exact_match: bool

# 1-5. [추천 API] 최종 응답
class RecommendationResponse(BaseModel):
    companies: List[CompanyInfo]
    ai_report: str

# --- [요청] 클라이언트 -> AI (Request DTO) ---

# 2-1. 분석 요청 (텍스트만 줌)
class ResumeRequest(BaseModel):
    resume_text: str

# 2-2. 추천 요청 (JSON 구조체 줌)
class ResumeInputDTO(BaseModel):
    id: str
    target_role: str
    resume_content: Dict[str, Any]
    evaluation: Optional[EvaluationResult] = None  # 추가됨: 분석 API에서 생성된 평가 정보
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "TEST_USER_001",
                    "target_role": "Backend Developer",
                    "resume_content": {
                        "education": [
                            {"degree": "Bachelor", "major": "Computer Science", "status": "Graduated"}
                        ],
                        "skills": {
                            "essential": ["Python", "FastAPI", "SQL"],
                            "additional": ["Docker", "AWS"]
                        },
                        "professional_experience": [
                            {
                                "company": "Tech Corp",
                                "period": "24개월",
                                "role": "Software Engineer",
                                "key_tasks": ["API Development", "Database Design"]
                            }
                        ]
                    },
                    "evaluation": {
                        "grade": "B",
                        "score": 75,
                        "summary": "안정적인 기술 스택을 보유한 백엔드 개발자입니다.",
                        "pros": ["주요 기술 스택 숙련도"],
                        "cons": ["대규모 트래픽 처리 경험 부족"],
                        "recommended_companies": []
                    }
                }
            ]
        }
    }
