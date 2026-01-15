from fastapi import FastAPI, HTTPException
from typing import List
import json
from pathlib import Path

# 모듈 import
from app.schemas import Resume, MatchResult, Company
from app.services.matching_engine import MatchingEngine

app = FastAPI(
    title="AI Resume Matching System (Hybrid RAG)",
    description="S-BERT 벡터 검색 + 룰 베이스 하이브리드 엔진 적용",
    version="3.0"
)

# --- 데이터 로드 (API 정보 제공용) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

COMPANY_POOL = []
RESUME_DB = {}

def load_basic_data():
    global COMPANY_POOL, RESUME_DB
    try:
        # 기업 정보 (상세 검색은 엔진이 하지만, 전체 목록 조회용으로 로드)
        company_path = DATA_DIR / "company_50_pool.json"
        if company_path.exists():
            with open(company_path, 'r', encoding='utf-8') as f:
                c_data = json.load(f)
                COMPANY_POOL = [Company(**c) for c in c_data]
        
        # 이력서 정보
        resume_path = DATA_DIR / "final_resume_600.json"
        if resume_path.exists():
            with open(resume_path, 'r', encoding='utf-8') as f:
                r_data = json.load(f)
                RESUME_DB = {r['id']: Resume(**r) for r in r_data}
                
        print(f"✅ 기본 데이터 로드 완료: Resume {len(RESUME_DB)}개")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")

load_basic_data()

# 엔진 초기화 (여기서 벡터 데이터 등을 로드함)
engine = MatchingEngine()

@app.get("/")
def read_root():
    return {"status": "ok", "engine": "Hybrid RAG Engine Active"}

@app.get("/companies", response_model=List[Company])
def get_companies():
    return COMPANY_POOL

@app.get("/resumes/{resume_id}", response_model=Resume)
def get_resume(resume_id: str):
    if resume_id not in RESUME_DB:
        raise HTTPException(status_code=404, detail="Resume not found")
    return RESUME_DB[resume_id]

@app.post("/match/{resume_id}", response_model=List[MatchResult])
def match_resume(resume_id: str):
    """
    Hybrid Engine을 사용하여 정교한 기업 추천 수행
    """
    if resume_id not in RESUME_DB:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    resume = RESUME_DB[resume_id]
    
    # 1. 엔진에 이력서 데이터 전달 (Pydantic -> Dict 변환)
    # 엔진이 내부적으로 resume_content, target_role 등을 사용함
    resume_dict = resume.dict()
    
    # 2. 추천 알고리즘 실행
    recommendations, ai_report = engine.recommend(resume_dict)
    
    # 3. 결과 매핑 (Engine Output -> API Schema)
    results = []
    for rec in recommendations:
        metadata = rec['metadata']
        score = rec['raw_score']
        
        # 매칭 타입 결정 (점수 구간별)
        if score >= 88: match_type = "🏆 Best Match"
        elif score >= 78: match_type = "✅ High Fit"
        else: match_type = "⚠️ Skill Gap"
        
        # 상세 사유 구성
        reason_detail = []
        if rec['is_exact_match']: reason_detail.append("직무 일치")
        if rec.get('is_related_role'): reason_detail.append("연관 직무(Flexible)")
        reason_detail.append(f"AI 적합도 {score}%")
        
        # AI Report가 있으면 첫 번째 결과에만 붙여주거나, 별도 필드로 제공
        # 여기서는 reason 필드에 요약해서 넣음
        
        results.append(MatchResult(
            company_name=metadata.get('name') or metadata.get('company_name'),
            match_score=score,
            tier=metadata.get('tier', 'Unknown'),
            match_type=match_type,
            reason=", ".join(reason_detail)
        ))
    
    return results