import json
import asyncio
import httpx
import os
from typing import Dict, Any
from dotenv import load_dotenv
from app.dto import ResumeAnalysisResponse

# .env 파일 로드
load_dotenv()

class AIAnalyzer:
    def __init__(self):
        # 환경 변수에서 OpenAI API Key 로드
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_url = "https://api.openai.com/v1/chat/completions"
        self.model_name = "gpt-4o-mini"
        
        if not self.api_key:
            print("⚠️ [Warning] .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        else:
            print(f"[AI Analyzer] Initialized with {self.model_name} Engine (API Key Loaded)")

    async def _call_openai_with_retry(self, system_prompt: str, user_prompt: str, retries: int = 5):
        """에러 500/422 및 API 전송 실패 시 지수 백오프 재시도 로직"""
        if not self.api_key:
            print("❌ [Error] API Key가 없습니다. 요청을 보낼 수 없습니다.")
            return None

        delay = 1
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for i in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    payload = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1 # 선생님께서 추후 수정하실 수 있도록 기본값 유지
                    }
                    
                    response = await client.post(self.model_url, headers=headers, json=payload, timeout=45.0)
                    
                    if response.status_code == 200:
                        return response.json()
                    
                    print(f"⚠️ OpenAI API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"⚠️ Connection Error: {e}")
            
            await asyncio.sleep(delay)
            delay *= 2
            
        return None

    async def analyze_resume(self, resume_text: str) -> ResumeAnalysisResponse:
        """이력서 텍스트를 GPT-4o-mini로 분석하여 구조화된 JSON 반환"""
        
        system_instruction = """
        당신은 전문 리크루팅 AI입니다. 반드시 JSON 형식으로만 응답하며, 다음 구조를 지키세요:
        {
            "classification": {
                "predicted_role": "backend | frontend | pm | fullstack | da 중 선택",
                "keywords": ["기술스택1", "기술스택2", ...]
            },
            "evaluation": {
                "grade": "S | A | B | C | F 중 선택",
                "score": 0~100 사이 숫자,
                "summary": "전체 요약문",
                "pros": ["강점1", "강점2"],
                "cons": ["보완점1", "보완점2"],
                "reasoning": "점수 산출 근거 (ATS 공식 $S_{matched} / S_{required}$ 언급 포함)",
                "recommended_companies": ["기업명1", "기업명2"]
            }
        }
        등급 기준: S(90+), A(80+), B(70+), C(60+), F(60미만)
        """
        
        user_prompt = f"다음 이력서를 분석하여 JSON으로 출력하세요:\n\n{resume_text}"
        
        result = await self._call_openai_with_retry(system_instruction, user_prompt)
        
        if not result:
            return self._get_fallback()

        try:
            content_text = result['choices'][0]['message']['content']
            raw_data = json.loads(content_text)
            
            cleaned_data = self._recover_data(raw_data)
            return ResumeAnalysisResponse(**cleaned_data)
            
        except Exception as e:
            print(f"❌ JSON Parsing Error: {e}")
            return self._get_fallback()

    def _recover_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """필수 필드 누락 시 기본값으로 채워주는 방어 코드"""
        if "evaluation" not in data:
            data["evaluation"] = {}
        
        eval_p = data["evaluation"]
        if not eval_p.get("pros"): eval_p["pros"] = ["경력 사항의 구체성"]
        if not eval_p.get("cons"): eval_p["cons"] = ["정량적 수치 보완 필요"]
        
        return data

    def _get_fallback(self) -> ResumeAnalysisResponse:
        """분석 실패 시 반환할 안전한 기본값"""
        return ResumeAnalysisResponse(
            classification={"predicted_role": "미분류", "keywords": []},  # pyright: ignore[reportArgumentType]
            evaluation={  # pyright: ignore[reportArgumentType]
                "grade": "C", 
                "score": 60, 
                "summary": "AI 서버 응답 지연으로 인한 기본 분석 결과입니다.",
                "pros": ["이력서 텍스트 수신 확인"], 
                "cons": ["상세 분석 실패"], 
                "reasoning": "API 호출 실패로 인해 기본 점수를 부여하였습니다.",
                "recommended_companies": []
            }
        )

# 서비스 인스턴스 생성
ai_analyzer = AIAnalyzer()