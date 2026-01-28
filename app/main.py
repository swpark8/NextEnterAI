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
from services.interview_engine import InterviewEngine
from services.file_parser import FileParser  # ✅ Import FileParser

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

# ... (omitted code) ...

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
    
    # 그 외 어떤 필드가 들어와도 에러내지 않음
    class Config:
        extra = "ignore" 

# ... (omitted code) ...

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
        body_dict = await request.json()
        interview_request = InterviewRequest(**body_dict)

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
            interview_request.portfolio_files
        )

        response = {
            "status": "success",
            "resume_id": interview_request.id,
            "target_role": final_target_role,
            "realtime": realtime
        }
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Interview Engine Error: {str(e)}")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "NextEnter AI Server is running properly (Hybrid Mode)."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
