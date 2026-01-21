import uvicorn
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
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
    version="2.2.0 (Hybrid Mode)"
)

# CORS 설정 (React 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 엔진 초기화 (서버 시작 시 1회 로드)
# ==========================================
print("🚀 Server initializing...")
try:
    engine = MatchingEngine()
    print("✅ Server ready to accept requests.")
except Exception as e:
    print(f"⚠️ Engine Load Error: {e}")
    engine = None

# ==========================================
# 3. 데이터 모델 정의 (유연한 구조 적용)
# ==========================================

# (1) 요청 데이터 - [수정됨] 아주 관대한 모델 (Hybrid Request)
# 프론트엔드가 어떤 형식으로 보내든 일단 받아서 처리합니다.
class ResumeRequest(BaseModel):
    id: Optional[str] = "USER_TEMP"
    
    # 1. 필수였던 필드들을 Optional로 변경 (422 에러 방지)
    target_role: Optional[str] = Field(None, description="희망 직무")
    
    # 2. 신규 구조 (Nested)
    resume_content: Optional[Dict[str, Any]] = None
    
    # 3. 구형 구조 (Flat) - 낱개로 들어올 경우를 대비
    education: Optional[List[Any]] = None
    skills: Optional[Any] = None # Dict or List
    professional_experience: Optional[List[Any]] = None
    project_experience: Optional[List[Any]] = None
    
    # 그 외 어떤 필드가 들어와도 에러내지 않음
    class Config:
        extra = "ignore" 

# (2) 응답 데이터 구조 (변경 없음)
class CompanyRecommendation(BaseModel):
    company_name: str
    match_score: float
    tier: str
    match_type: str
    match_level: str
    reason: str
    tech_stack: List[str]
    missing_skills: List[str]
    keyword_raw: float
    vector_norm: float
    ats_score: Optional[Dict[str, Any]] = None
    raw_score: float
    is_exact_match: bool
    metadata: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    status: str = "success"
    resume_id: str
    target_role: str
    grade: str
    score: float
    ai_feedback: str
    recommendations: List[CompanyRecommendation]

# ==========================================
# 4. Exception Handler (Pydantic 검증 에러 상세 처리)
# ==========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Pydantic 검증 에러 발생 시 상세한 에러 메시지 반환
    """
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg"),
            "type": error.get("type"),
            "input": error.get("input")
        })
    
    print(f"❌ [Validation Error] Request URL: {request.url}")
    print(f"❌ [Validation Error] Request Method: {request.method}")
    print(f"❌ [Validation Error] Errors: {json.dumps(error_details, indent=2, ensure_ascii=False)}")
    
    # 요청 본문 로깅 (가능한 경우)
    # 주의: RequestValidationError 발생 시 본문이 이미 소비되었을 수 있음
    try:
        # Starlette의 Request는 body를 한 번만 읽을 수 있으므로,
        # ValidationError 발생 시 이미 소비되었을 수 있음
        body = await request.body()
        if body:
            print(f"❌ [Validation Error] Request Body: {body.decode('utf-8')}")
        else:
            print(f"⚠️ [Validation Error] Request body가 비어있거나 이미 소비되었습니다.")
    except Exception as e:
        # 본문이 이미 소비되었거나 읽을 수 없는 경우는 정상일 수 있음
        print(f"⚠️ [Validation Error] Request body 읽기 실패 (이미 소비되었을 수 있음): {e}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": error_details,
            "message": "요청 데이터 검증 실패",
            "errors": error_details
        }
    )

# ==========================================
# 5. API 엔드포인트
# ==========================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_resume(request: Request):  # ← 일단 raw Request로 받기
    """
    디버깅용: 실제 들어오는 body를 먼저 확인
    """
    try:
        # 1. Raw body 확인
        raw_body = await request.body()
        print(f"🔍 [Raw Body] {raw_body.decode('utf-8')}")
        
        # 2. JSON 파싱
        body_dict = await request.json()
        print(f"🔍 [Parsed JSON] {json.dumps(body_dict, indent=2, ensure_ascii=False)}")
        
        # 3. Pydantic 모델로 변환
        resume_request = ResumeRequest(**body_dict)
        print(f"🔍 [Pydantic Model] {resume_request}")
        
        # 4. 기존 로직 실행
        request_obj = resume_request  # 이름 변경
        
        final_target_role = request_obj.target_role
        if not final_target_role:
            print("⚠️ 'target_role'이 비어있습니다. 기본값 'backend'로 설정합니다.")
            final_target_role = "backend"

        final_content = request_obj.resume_content
        if not final_content:
            print("⚠️ 'resume_content' (포장 상자)가 없습니다. 낱개 데이터를 조립합니다.")
            final_content = {
                "education": request_obj.education or [],
                "skills": request_obj.skills or {},
                "professional_experience": request_obj.professional_experience or [],
                "project_experience": request_obj.project_experience or []
            }
        
        resume_input = {
            "id": request_obj.id,
            "target_role": final_target_role,
            "resume_content": final_content,
            "classification": {},
            "evaluation": {}
        }
        
        print(f"🔍 Analyzing for role: {final_target_role}")

        if engine:
            results, report = engine.recommend(resume_input)
        else:
            raise Exception("Engine not initialized")
        
        if not results:
            print("⚠️ No recommendations generated.")
            grade = "F"
            top_score = 0.0
        else:
            top_score = results[0]['match_score']
            grade = engine.get_grade(top_score)

        response = {
            "status": "success",
            "resume_id": request_obj.id,
            "target_role": final_target_role,
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
        raise HTTPException(status_code=500, detail=f"Server Logic Error: {str(e)}")
    
# [Legacy Alias]
@app.post("/api/v1/recommend", response_model=AnalysisResponse, tags=["Legacy"])
async def recommend_resume_alias(resume_request: ResumeRequest):
    """
    /recommend 요청도 위와 똑같이 처리합니다.
    """
    print("🔄 Redirecting /recommend to /analyze...")
    return await analyze_resume(resume_request)

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "NextEnter AI Server is running properly (Hybrid Mode)."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)