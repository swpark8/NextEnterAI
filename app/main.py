from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import json
import os
from pathlib import Path

app = FastAPI(title="NextEnter AI API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "NextEnter AI API is running"}

# ==========================================
# JSON 데이터 테스트 함수 (개발/테스트용)
# ==========================================
def test_matching_with_json(limit=3):
    """
    data 폴더의 final_resume_600.json을 읽어서 
    매칭 엔진이 정상 동작하는지 테스트합니다.
    """
    from app.services.matching_engine import MatchingEngine
    
    print("\n[Test] Starting JSON matching test...")
    engine = MatchingEngine()
    
    # 테스트용 이력서 데이터 경로
    resume_path = engine.base_path / "final_resume_600.json"
    
    if not resume_path.exists():
        print(f"[Error] Test data not found: {resume_path}")
        return

    with open(resume_path, 'r', encoding='utf-8') as f:
        resumes = json.load(f)
    
    print(f"[Info] Testing top {limit} out of {len(resumes)} resumes.")
    
    for i, resume in enumerate(resumes[:limit]):
        print(f"\n--- [Test {i+1}] Resume ID: {resume.get('id')} ---")
        print(f"Target Role: {resume.get('target_role')}")
        
        try:
            recommendations, ai_report = engine.recommend(resume)
            
            print(f"Recommended Companies ({len(recommendations)}):")
            for rec in recommendations:
                c_meta = rec['metadata']
                name = c_meta.get('company_name') or c_meta.get('name')
                score = rec.get('raw_score')
                print(f"   - {name} (Score: {score})")
            
            print(f"AI Report Summary:\n{ai_report[:150]}...")
            
        except Exception as e:
            print(f"[Error] Exception during test: {e}")

if __name__ == "__main__":
    # 스크립트로 직접 실행할 경우 테스트 함수 호출
    import sys
    
    # 만약 --test 인자가 있으면 테스트 실행
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_matching_with_json()
    else:
        # 기본적으로는 서버 실행 안내 출력
        print("\n[Run] NextEnter AI Server:")
        print("uvicorn app.main:app --reload")
        print("\n[Test] Matching Engine Test:")
        print("python -m app.main --test")
