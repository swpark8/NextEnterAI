# AI Analysis service
import json
from typing import Dict, Any
from app.dto import ResumeAnalysisResponse

class AIAnalyzer:
    def __init__(self):
        print("🤖 AI 분석기(Analyzer) 초기화 완료 (데이터 복구 모드 켜짐)")

    def analyze_resume(self, resume_text: str) -> ResumeAnalysisResponse:
        """
        이력서 텍스트를 받아서 분석 결과(JSON)를 반환합니다.
        (현재는 시뮬레이션 데이터 반환)
        """
        # (여기에 실제 LLM 호출 로직이 들어갑니다.)
        
        try:
            # === [시뮬레이션] 가짜 데이터 ===
            mock_response = {
                "classification": {
                    "predicted_role": "Senior Java Developer",
                    "keywords": ["Java", "Spring Boot", "Kafka"]
                },
                "evaluation": {
                    "grade": "A",
                    "score": 88,
                    "summary": "MSA 경험이 풍부한 백엔드 개발자입니다.",
                    # pros/cons 누락 상황 가정
                    "reasoning": "대규모 트래픽 처리 경험과 MSA 설계 능력이 매우 뛰어납니다.",
                    "recommended_companies": ["Naver", "Line"]
                }
            }
            
            # 데이터 복구 (누락된 pros/cons 채우기)
            cleaned_data = self._recover_data(mock_response)
            return ResumeAnalysisResponse(**cleaned_data)

        except Exception as e:
            print(f"분석 에러: {e}")
            return self._get_fallback()

    def _recover_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """장점/단점이 없으면 reasoning에서 유추해서 채워넣는 함수"""
        eval_data = data.get("evaluation", {})
        
        if not eval_data.get("pros"):
            # 간단한 채움 로직
            eval_data["pros"] = ["탄탄한 기술 스택 보유", "관련 실무 경험 풍부"]
            
        if not eval_data.get("cons"):
            eval_data["cons"] = ["클라우드 네이티브 기술 학습 권장", "정량적 성과 추가 기술 필요"]
            
        data["evaluation"] = eval_data
        return data

    def _get_fallback(self):
        """에러 시 기본값 반환"""
        return {
            "classification": {"predicted_role": "Unknown", "keywords": []},
            "evaluation": {
                "grade": "F", "score": 0, "summary": "분석 실패",
                "pros": [], "cons": [], "recommended_companies": []
            }
        }
ai_analyzer = AIAnalyzer()
