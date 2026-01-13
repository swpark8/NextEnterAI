# Matching enimport os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# 라이브러리 없어도 서버가 죽지 않게 처리 (설치 안내용)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
except ImportError:
    SentenceTransformer = None
    OpenAI = None

class MatchingEngine:
    """
    [Hybrid RAG Engine]
    벡터 검색(S-BERT) + 정밀 리랭킹(Weighted Scoring) + AI 리포트(GPT)
    """
    
    def __init__(self, base_path: str = "./data", openai_api_key: str = None):
        print("🚀 매칭 엔진(Matching Engine) 초기화 중...")
        
        if not SentenceTransformer:
            print("⚠️ 경고: sentence_transformers 라이브러리가 없습니다.")
            return

        self.base_path = base_path
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # 모델 로드 (시간이 좀 걸림, 실제 배포 시 주석 해제)
        # self.model = SentenceTransformer('jhgan/ko-sroberta-multitask') 
        self.model = None 
        print("✅ 매칭 엔진 준비 완료")

    def recommend(self, resume_data: Dict[str, Any]) -> Tuple[List[Dict], str]:
        """
        이력서 데이터를 받아 추천 기업 리스트와 AI 리포트를 반환
        """
        # --- [로직 이식] ---
        # 실제로는 여기서 벡터 검색과 리랭킹이 일어납니다.
        # 지금은 테스트를 위해 가짜(Mock) 데이터를 반환하도록 설정합니다.
        
        recommended_companies = [
            {
                "metadata": {"company_name": "네이버 (Naver)", "job_title": "Backend Dev"},
                "raw_score": 92.5,
                "is_exact_match": True
            },
            {
                "metadata": {"company_name": "토스 (Toss)", "job_title": "Server Engineer"},
                "raw_score": 88.0,
                "is_exact_match": True
            },
            {
                "metadata": {"company_name": "당근마켓", "job_title": "Platform Dev"},
                "raw_score": 85.3,
                "is_exact_match": False
            }
        ]
        
        ai_report = """
        [AI 컨설팅 리포트]
        지원자님의 기술 스택(Java/Spring)은 네이버와 토스의 요구사항과 90% 이상 일치합니다.
        특히 대용량 트래픽 처리 경험이 돋보입니다. 
        다만, 당근마켓 지원을 위해서는 Go 언어에 대한 추가 학습이 도움이 될 것입니다.
        """
        
        return recommended_companies, ai_report

# 싱글톤 인스턴스 생성 (API 키는 환경변수에서 가져오세요)
matching_engine = MatchingEngine()
