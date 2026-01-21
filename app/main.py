import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# [핵심] 우리가 만든 엔진 임포트
# (파일명이 resume_engine.py 라고 가정)
from services.resume_engine import MatchingEngine

# ==========================================
# 1. FastAPI 앱 설정
# ==========================================
app = FastAPI(
    title="NextEnter AI Resume Analysis Server",
    description="이력서 평가 및 기업 추천 AI 엔진 API",
    version="2.1.0"
)

# CORS 설정 (React 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안을 위해 배포 시에는 구체적 도메인 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 엔진 초기화 (서버 시작 시 1회 로드)
# ==========================================
print("🚀 Server initializing...")
engine = MatchingEngine()
print("✅ Server ready to accept requests.")

# ==========================================
# 3. 데이터 모델 정의 (Pydantic) - Schema 통합 완료
# ==========================================

# (1) 요청 데이터 (React -> Python)
# resume.py의 모든 필드를 수용하도록 설계됨
class ResumeRequest(BaseModel):
    # 필수 필드
    id: Optional[str] = "USER_TEMP"
    target_role: str = Field(..., description="희망 직무 (backend, frontend, pm 등)")
    
    # [복구] resume.py에 있던 선택 필드들 완벽 이식
    candidate_id: Optional[str] = None
    standardized_role: Optional[Dict[str, Any]] = None
    
    # [핵심 전략] 하위 객체(Education, Skills 등)를 Dict로 통합하여 422 에러 원천 차단
    # 기존 ResumeContent 클래스 내용을 이 Dict 안에 모두 담습니다.
    resume_content: Dict[str, Any] = Field(..., description="이력서 상세 (학력, 스킬, 경력 포함)")
    
    # 추가 메타데이터
    classification: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None

# (2) 응답 데이터 - 추천 기업 상세 정보
# resume_engine_fixed.py가 뱉는 결과물과 1:1 매칭
class CompanyRecommendation(BaseModel):
    # 기본 정보
    company_name: str
    match_score: float
    tier: str
    match_type: str
    match_level: str
    reason: str
    tech_stack: List[str]
    missing_skills: List[str]
    
    # 상세 점수 (피드백 생성용)
    keyword_raw: float
    vector_norm: float
    ats_score: Optional[Dict[str, Any]] = None
    
    # [Legacy 호환] 기존 프론트엔드 코드 깨짐 방지
    raw_score: float
    is_exact_match: bool
    
    # 메타데이터 (UI 표시용)
    metadata: Optional[Dict[str, Any]] = None

# (3) 최종 API 응답 구조
class AnalysisResponse(BaseModel):
    status: str = "success"
    resume_id: str
    target_role: str
    
    # 분석 결과
    grade: str
    score: float
    ai_feedback: str  # XAI 리포트
    
    # 추천 리스트
    recommendations: List[CompanyRecommendation]

# ==========================================
# 4. API 엔드포인트
# ==========================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_resume(request: ResumeRequest):
    """
    [Main API] 이력서를 받아 분석하고 추천 기업과 피드백을 반환합니다.
    """
    try:
        print(f"📥 [Request] Analyzing resume: {request.id} ({request.target_role})")
        
        # 1. 요청 데이터를 딕셔너리로 변환 (엔진 입력용)
        # Pydantic 모델을 dict로 바꾸면 엔진이 쓰기 편함
        resume_input = request.model_dump()
        
        # 2. 엔진 실행
        # recommend 함수는 (formatted_results, ai_report_string) 튜플을 반환함
        results, report = engine.recommend(resume_input)
        
        # 3. 데이터 검증 및 안전장치
        if not results:
            print("⚠️ No recommendations generated.")
            grade = "F"
            top_score = 0.0
        else:
            # 1위 기업 점수 기반으로 등급 표시
            top_score = results[0]['match_score']
            grade = engine.get_grade(top_score)

        # 4. 응답 생성
        response = {
            "status": "success",
            "resume_id": request.id,
            "target_role": request.target_role,
            "grade": grade,
            "score": top_score,
            "ai_feedback": report,
            "recommendations": results
        }
        
        print(f"📤 [Response] Success! Grade: {grade}, Recs: {len(results)}")
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ [Error] {str(e)}")
        # 422 Validation Error가 아닌 500 내부 에러로 명확히 반환
        raise HTTPException(status_code=500, detail=f"Server Logic Error: {str(e)}")
    
# [추가됨] 사용자 안심용 Legacy Alias
@app.post("/api/v1/recommend", response_model=AnalysisResponse, tags=["Legacy"])
async def recommend_resume_alias(request: ResumeRequest):
    """
    [Alias] /api/v1/analyze 와 동일하게 동작합니다.
    (기존 recommend API를 찾는 사용자를 위한 별칭)
    """
    print("🔄 Redirecting /recommend to /analyze...")
    return await analyze_resume(request)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "NextEnter AI Server is running properly."}

# ==========================================
# 5. 서버 실행 (직접 실행 시)
# ==========================================
if __name__ == "__main__":
    # 포트 8000번에서 실행 (React는 보통 3000번)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)