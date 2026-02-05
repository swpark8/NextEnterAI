import uvicorn
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Union

# [핵심] 우리가 만든 엔진 임포트
# (파일명이 resume_engine.py 라고 가정)
from app.services.resume_engine import MatchingEngine
from app.services.interview_engine import InterviewEngine
from app.services.file_parser import FileParser  # ✅ Import FileParser

# ==========================================
# 1. FastAPI 앱 설정
# ==========================================
app = FastAPI(
    title="NextEnter AI Resume Analysis Server",
    description="이력서 평가 및 기업 추천 AI 엔진 API",
    version="2.3.0 (File Parser Integrated)"
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
# 2. 엔진 인스턴스 및 세션 관리
# ==========================================

# Resume Engine 초기화
engine = MatchingEngine()

# Interview Engine 세션 관리 (메모리 기반)
interview_engines: Dict[str, InterviewEngine] = {}

def get_interview_engine(user_id: str) -> InterviewEngine:
    """사용자별 면접 엔진 인스턴스 반환 (세션 관리)"""
    if user_id not in interview_engines:
        interview_engines[user_id] = InterviewEngine()
    return interview_engines[user_id]

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
    raw_text: Optional[str] = None
    file_path: Optional[str] = None  # ✅ 파일 경로 필드 추가
    
    # 3. 구형 구조 (Flat) - 낱개로 들어올 경우를 대비
    education: Optional[List[Any]] = None
    skills: Optional[Any] = None # Dict or List
    professional_experience: Optional[List[Any]] = None
    project_experience: Optional[List[Any]] = None
    
    # 4. Java 백엔드 연동 (등급/분류)
    classification: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    
    # 그 외 어떤 필드가 들어와도 에러내지 않음
    class Config:
        extra = "ignore" 

# Interview 요청 모델
class InterviewRequest(BaseModel):
    id: Optional[str] = "USER_TEMP"
    target_role: Optional[str] = None
    resume_content: Optional[Dict[str, Any]] = None
    last_answer: Optional[str] = None
    portfolio: Optional[Dict[str, Any]] = None
    portfolio_files: Optional[List[str]] = None
    classification: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    education: Optional[List[Any]] = None
    skills: Optional[Any] = None
    professional_experience: Optional[List[Any]] = None
    project_experience: Optional[List[Any]] = None
    total_turns: Optional[int] = 5  # ✅ 전체 면접 질문 횟수 추가
    
    # [NEW] Stateless Context Support
    chat_history: Optional[List[Dict[str, Any]]] = None # 이전 대화 내용 (Stateless 지원)
    difficulty: Optional[str] = "JUNIOR" # JUNIOR | SENIOR
    
    class Config:
        extra = "ignore"

# Interview 응답 모델
class InterviewResponse(BaseModel):
    status: str
    resume_id: str
    target_role: str
    realtime: Dict[str, Any]

# Analysis 응답 모델
class AnalysisResponse(BaseModel):
    status: str
    resume_id: str
    target_role: str
    grade: str
    score: float
    ai_feedback: Any
    recommendations: List[Any]

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_resume(request: Union[Request, ResumeRequest]):
    """
    이력서 분석 및 AI 기업 매칭. Request(직접 호출) 또는 ResumeRequest(/recommend 경유) 모두 처리.
    """
    try:
        # 1. 인자 분기: Request면 body 파싱, ResumeRequest면 그대로 사용
        if isinstance(request, Request):
            raw_body = await request.body()
            print(f"🔍 [Raw Body] {raw_body.decode('utf-8')}")
            body_dict = await request.json()
            print(f"🔍 [Parsed JSON] {json.dumps(body_dict, indent=2, ensure_ascii=False)}")
            request_obj = ResumeRequest(**body_dict)
            print(f"🔍 [Pydantic Model] {request_obj}")
        else:
            request_obj = request  # /recommend에서 넘어온 ResumeRequest
            print(f"🔍 [Parsed] ResumeRequest (from /recommend)")

        # ✅ [New] 파일 파싱 로직 추가 (이력서 파일이 있으면 텍스트 추출)
        if request_obj.file_path:
            print(f"📂 Parsing resume file from: {request_obj.file_path}")
            extracted_text = FileParser.parse_file(request_obj.file_path)
            
            if extracted_text and not extracted_text.startswith("[Error]"):
                print(f"✅ Extracted {len(extracted_text)} chars from file.")
                # raw_text에 추가 (기존 텍스트가 있다면 병합)
                existing_text = request_obj.raw_text or ""
                request_obj.raw_text = existing_text + "\n\n[Parsed File Content]\n" + extracted_text
            else:
                print(f"⚠️ File parsing failed or file empty: {extracted_text}")
        
        final_target_role = request_obj.target_role
        if not final_target_role:
            print("⚠️ 'target_role'이 비어있습니다. 기본값 'backend'로 설정합니다.")
            final_target_role = "backend"

        final_content = request_obj.resume_content
        if not final_content:
            print("⚠️ 'resume_content' (포장 상자)가 없습니다. 낱개 데이터를 조립합니다.")
            final_content = {
                "raw_text": request_obj.raw_text,
                "education": request_obj.education or [],
                "skills": request_obj.skills or {},
                "professional_experience": request_obj.professional_experience or [],
                "project_experience": request_obj.project_experience or []
            }
        else:
            # resume_content가 이미 있지만, raw_text 가 업데이트 되었을 수 있으므로 동기화
            if request_obj.raw_text:
                if "raw_text" in final_content:
                    final_content["raw_text"] += "\n\n" + request_obj.raw_text
                else:
                    final_content["raw_text"] = request_obj.raw_text
        
        resume_input = {
            "id": request_obj.id,
            "target_role": final_target_role,
            "resume_content": final_content,
            "classification": (request_obj.classification or {}),
            "evaluation": (request_obj.evaluation or {})
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
            
            # [FIX] Java에서 받은 등급 정보 우선 사용
            evaluation = resume_input.get('evaluation', {})
            grade = evaluation.get('grade')
            
            if grade:
                print(f"✅ [등급 정보] Java에서 받은 등급 사용: {grade}")
            else:
                # 등급 정보가 없으면 자동 계산
                grade = engine.get_grade(top_score)
                print(f"⚠️ [등급 정보] 자동 계산된 등급 사용: {grade}")

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
        # print(f"❌ [Error] {str(e)}") # traceback에서 출력됨
        raise HTTPException(status_code=500, detail=f"Server Logic Error: {str(e)}")
    
# [Legacy Alias]
@app.post("/api/v1/recommend", response_model=AnalysisResponse, tags=["Legacy"])
async def recommend_resume_alias(resume_request: ResumeRequest):
    """
    /recommend 요청도 위와 똑같이 처리합니다.
    """
    print("🔄 Redirecting /recommend to /analyze...")
    return await analyze_resume(resume_request)

@app.post("/api/v1/interview/next", response_model=InterviewResponse)
async def interview_next(request: Request):
    try:
        # 1. Robust Body Parsing (Fix for Empty Body / Debugging)
        body_bytes = await request.body()
        if not body_bytes:
             print("❌ [Error] Received 0 bytes body in /interview/next")
             raise HTTPException(status_code=400, detail="Empty request body received")
        
        try:
            body_json = await request.json()
            print(f"🔍 [Interview Request] Raw JSON: {json.dumps(body_json, indent=None, ensure_ascii=False)[:300]}...") 
        except Exception as e:
            # Enhanced error logging for debugging encoding issues
            print(f"❌ [Error] JSON Parse Failed: {str(e)}")
            try:
                # Try to decode with replacement to show what we received
                print(f"❌ [Error] Body Preview (lossy): {body_bytes.decode('utf-8', errors='replace')[:500]}")
            except:
                pass
            raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

        # 2. Convert to Pydantic Model manually
        try:
            interview_request = InterviewRequest(**body_json)
        except ValidationError as ve:
            print(f"❌ [Error] Pydantic Validation Failed: {ve}")
            raise HTTPException(status_code=422, detail=f"Validation Error: {ve}")

        # 3. Core Logic
        print(f"🔍 [Interview Logic] id={interview_request.id}, role={interview_request.target_role}")

        final_target_role = interview_request.target_role
        if not final_target_role and interview_request.classification:
            final_target_role = interview_request.classification.get("predicted_role")
        if not final_target_role:
            final_target_role = "backend"

        final_content = interview_request.resume_content
        if not final_content:
            final_content = {
                "education": interview_request.education or [],
                "skills": interview_request.skills or {},
                "professional_experience": interview_request.professional_experience or [],
                "project_experience": interview_request.project_experience or []
            }
        # raw_text fallback: 구조화 필드가 비어 있으면 raw_text를 요약용으로 유지 (면접 엔진에서 사용)
        if final_content and final_content.get("raw_text"):
            sk = final_content.get("skills")
            skills_nonempty = (
                (isinstance(sk, list) and len(sk) > 0)
                or (isinstance(sk, dict) and (len(sk.get("essential") or []) > 0 or len(sk.get("additional") or []) > 0))
            )
            has_structure = (
                skills_nonempty
                or (isinstance(final_content.get("education"), list) and len(final_content.get("education", [])) > 0)
                or (isinstance(final_content.get("professional_experience"), list) and len(final_content.get("professional_experience", [])) > 0)
                or (isinstance(final_content.get("project_experience"), list) and len(final_content.get("project_experience", [])) > 0)
            )
            if not has_structure:
                final_content["_raw_text_primary"] = True  # 엔진에서 raw_text를 우선 사용

        resume_input = {
            "id": interview_request.id,
            "target_role": final_target_role,
            "resume_content": final_content,
            "classification": interview_request.classification or {},
            "evaluation": interview_request.evaluation or {}
        }

        # 세션별 엔진 인스턴스 획득
        itv_engine = get_interview_engine(interview_request.id)

        realtime = itv_engine.generate_response(
            resume_input,
            interview_request.portfolio,
            interview_request.last_answer,
            interview_request.portfolio_files,
            total_turns=interview_request.total_turns, # ✅ total_turns 전달
            chat_history=interview_request.chat_history, # ✅ [NEW] Chat History 전달
            difficulty=interview_request.difficulty # ✅ [NEW] Difficulty 전달
        )

        response = {
            "status": "success",
            "resume_id": interview_request.id,
            "target_role": final_target_role,
            "realtime": realtime
        }
        print(f"📤 [Interview Response] Success for resume_id={interview_request.id}")
        return response


    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Interview Engine Error: {str(e)}")

class FinalizeRequest(BaseModel):
    id: str
    chat_history: Optional[List[Dict[str, Any]]] = None # [NEW] Stateless 지원

@app.post("/api/v1/interview/finalize")
async def interview_finalize(request: FinalizeRequest):
    """
    [POST] /api/v1/interview/finalize
    면접을 종료하고 최종 평가 리포트를 반환합니다.
    """
    try:
        print(f"🏁 Finalizing interview for ID: {request.id}")
        
        # 1. 엔진 인스턴스 조회
        if request.id not in interview_engines:
            raise HTTPException(status_code=404, detail="진행 중인 면접 세션이 없습니다.")
            
        itv_engine = get_interview_engine(request.id)
        
        # 2. 리포트 생성
        result = itv_engine.finalize_interview(chat_history=request.chat_history)
        
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
             
        # 3. 세션 정리 (선택 사항: 리포트 생성 후 세션을 유지할지 삭제할지 결정. 여기서는 유지)
        # del interview_engines[request.id] 
        
        print(f"✅ Final Report Generated: {result.get('result')}, Score: {result.get('total_score')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [Error] Finalize failed: {e}")
        raise HTTPException(status_code=500, detail=f"Finalize Error: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "NextEnter AI Server is running properly (Hybrid Mode)."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
