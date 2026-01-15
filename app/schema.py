from pydantic import BaseModel
from typing import List, Optional

# ==========================================
# 1. 이력서(Resume) 하위 모델들
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

# ==========================================
# 2. 메인 이력서 모델 (Resume)
# ==========================================

class Resume(BaseModel):
    id: str
    candidate_id: Optional[str] = None
    target_role: str
    standardized_role: StandardizedRole
    resume_content: ResumeContent
    resume_evaluation: Optional[ResumeEvaluation] = None

# ==========================================
# 3. 회사(Company) 모델
# ==========================================

class Company(BaseModel):
    name: str
    tier: str
    employee_count: Optional[str] = None
    industry: str
    location: str
    tech_stack: List[str]
    target_roles: List[str]

# ==========================================
# 4. API 응답 및 메타데이터 모델
# ==========================================

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