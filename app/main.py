# Main application entry point
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="NextEnterAI 엔진",
    description="AI 기반 이력서 분석 및 기업 추천 서비스 API",
    version="1.0.0"
)

# 라우터 연결 (주소 앞에 /api/v1 붙임)
app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health_check():
    """서버 상태 확인용"""
    return {"status": "healthy", "service": "NextEnterAI"}