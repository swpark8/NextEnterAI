from fastapi import APIRouter, UploadFile, File, HTTPException
from app.dto import (
    ResumeRequest, 
    ResumeAnalysisResponse, 
    RecommendationResponse, 
    ResumeInputDTO
)
# 각 서비스 모듈 가져오기 (Stub이나 구현된 파일이 있어야 함)
from app.services.ai_analyzer import ai_analyzer
from app.services.resume_validation_engine import resume_validation_engine
from app.core.text_processor import TextProcessor

# 라우터 객체 생성 (이게 있어야 main.py에서 import 가능!)
router = APIRouter()

# =================================================================
# 1. 이력서 분석 API (텍스트 -> JSON 분석)
# =================================================================
@router.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(request: ResumeRequest):
    """
    [POST] /api/v1/analyze
    이력서 텍스트를 입력받아 직무 분류 및 심층 평가 결과를 반환합니다.
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="이력서 내용이 비어있습니다.")
    
    return ai_analyzer.analyze_resume(request.resume_text)


# =================================================================
# 2. 기업 추천 API (이력서 JSON -> 기업 추천 & AI 리포트)
# =================================================================
@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_jobs(resume: ResumeInputDTO):
    """
    [POST] /api/v1/recommend
    이력서 정보(JSON)를 바탕으로 가장 적합한 기업 3곳을 추천하고 AI 리포트를 제공합니다.
    """
    try:
        companies, report = resume_validation_engine.recommend(resume.dict())
        
        # 프론트엔드용 데이터 변환
        formatted_companies = []
        for item in companies:
            formatted_companies.append({
                "company_name": item['metadata'].get('company_name', 'Unknown'),
                "role": item['metadata'].get('job_title', 'Unknown'),
                "score": round(item.get('raw_score', 0.0), 1),
                "is_exact_match": item.get('is_exact_match', False)
            })
            
        return {"companies": formatted_companies, "ai_report": report}
        
    except Exception as e:
        print(f"❌ [추천 API 에러] : {e}")
        raise HTTPException(status_code=500, detail="기업 매칭 중 오류가 발생했습니다.")


# =================================================================
# 3. 텍스트 추출 API (파일 업로드 -> 텍스트 변환)
# =================================================================
@router.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    [POST] /api/v1/extract-text
    PDF, DOCX, TXT 파일을 업로드하면 텍스트만 추출해서 반환합니다.
    """
    try:
        contents = await file.read()
        text = TextProcessor.extract_from_file(contents, file.filename or "")
        
        return {
            "filename": file.filename, 
            "extracted_text": text
        }
        
    except Exception as e:
        print(f"❌ [텍스트 추출 에러] : {e}")
        raise HTTPException(status_code=500, detail="파일 처리 중 오류가 발생했습니다.")