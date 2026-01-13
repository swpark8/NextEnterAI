import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
except ImportError:
    SentenceTransformer = None
    OpenAI = None
    cosine_similarity = None

class MatchingEngine:
    """
    [Hybrid RAG Engine]
    벡터 검색(S-BERT) + 정밀 리랭킹(Weighted Scoring) + AI 리포트(GPT)
    """
    
    def __init__(self, base_path: str = "./data", openai_api_key: str = None):
        print("🚀 매칭 엔진(Matching Engine) 초기화 중...")
        
        # 1. 라이브러리 체크
        if not SentenceTransformer:
            print("⚠️ [Critical] sentence_transformers 라이브러리가 없습니다.")
            self.model = None
            return

        # 2. OpenAI 클라이언트 설정
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        if not self.client:
            print("⚠️ [Warning] OpenAI API Key가 없습니다. AI 리포트 기능이 제한됩니다.")

        self.base_path = base_path
        
        # 3. 데이터 및 모델 로드
        try:
            print(f"📂 데이터 로딩 중... ({base_path})")
            
            # (1) 벡터 데이터 로드 (.npy)
            vector_path = os.path.join(base_path, "final_embedded_dataset.npy")
            if os.path.exists(vector_path):
                self.company_vectors = np.load(vector_path)
                print(f"  - 벡터 데이터 로드 완료: {self.company_vectors.shape}")
            else:
                print(f"❌ 벡터 파일을 찾을 수 없습니다: {vector_path}")
                self.company_vectors = None

            # (2) 메타데이터 로드 (.json)
            meta_path = os.path.join(base_path, "final_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.company_metadata = json.load(f)
                print(f"  - 메타데이터 로드 완료: {len(self.company_metadata)}개")
            else:
                print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {meta_path}")
                self.company_metadata = []

            # (3) 임베딩 모델 로드
            print("🧠 임베딩 모델 로딩 중 (jhgan/ko-sroberta-multitask)...")
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("✅ 모델 로드 완료")

        except Exception as e:
            print(f"❌ 초기화 중 에러 발생: {e}")
            self.model = None
            self.company_vectors = None

    def recommend(self, resume_data: Dict[str, Any]) -> Tuple[List[Dict], str]:
        """
        이력서 데이터를 받아 추천 기업 리스트와 AI 리포트를 반환
        규칙: 1, 2위는 Target Role 일치(Exact), 3위는 유연한 추천(Flexible)
        """
        if not self.model or self.company_vectors is None:
            return [], "⚠️ 매칭 엔진이 정상적으로 초기화되지 않았습니다."

        # 1. 이력서 텍스트 벡터화
        # 이력서의 주요 내용을 합쳐서 쿼리 생성
        resume_content = resume_data.get('resume_content', {})
        target_role = resume_data.get('target_role', '').lower()
        
        # 키워드와 요약을 합쳐서 임베딩
        eval_data = resume_content.get('evaluation', {})
        classification = resume_content.get('classification', {})
        
        query_text = f"{target_role} " + \
                     " ".join(classification.get('keywords', [])) + " " + \
                     eval_data.get('summary', '')
                     
        query_vector = self.model.encode(query_text)

        # 2. 코사인 유사도 계산
        # (1, 768) * (N, 768) -> (1, N)
        scores = cosine_similarity([query_vector], self.company_vectors)[0]

        # 3. Top N 후보군 추출 (넉넉하게 20개)
        top_n_indices = np.argsort(scores)[::-1][:20]
        
        candidates = []
        for idx in top_n_indices:
            company = self.company_metadata[idx]
            score = float(scores[idx]) * 100  # 100점 만점으로 변환
            
            # 직무 일치 여부 확인 (Target Role이 Job Title에 포함되는지)
            job_title = company.get('job_title', '').lower()
            is_exact_match = target_role in job_title if target_role else False
            
            candidates.append({
                "metadata": company,
                "raw_score": score,
                "is_exact_match": is_exact_match
            })

        # 4. 필터링 로직 (1,2위 Exact / 3위 Flexible)
        final_recommendations = self._apply_filtering_rules(candidates)

        # 5. AI 리포트 생성
        ai_report = self._generate_ai_report(resume_data, final_recommendations)
        
        return final_recommendations, ai_report

    def _apply_filtering_rules(self, candidates: List[Dict]) -> List[Dict]:
        """
        요구사항: 1, 2위는 Exact Match, 3위는 Flexible Match
        """
        exact_matches = [c for c in candidates if c['is_exact_match']]
        flexible_matches = [c for c in candidates if not c['is_exact_match']]
        
        result = []
        
        # 1위, 2위 선정 (Exact Match 우선)
        # Exact Match가 부족하면 Flexible에서 채움
        for _ in range(2):
            if exact_matches:
                result.append(exact_matches.pop(0))
            elif flexible_matches:
                result.append(flexible_matches.pop(0))
                
        # 3위 선정 (Flexible Match 우선)
        # Flexible이 없으면 Exact 남은 것에서 채움
        if flexible_matches:
            result.append(flexible_matches.pop(0))
        elif exact_matches:
            result.append(exact_matches.pop(0))
            
        return result

    def _generate_ai_report(self, resume_data: Dict, recommendations: List[Dict]) -> str:
        """
        OpenAI를 사용하여 개인화된 리포트 생성
        """
        if not self.client:
            return """
            [시스템 메시지]
            OpenAI API Key가 설정되지 않아 상세 AI 리포트를 생성할 수 없습니다.
            .env 파일을 확인해주세요.
            """

        try:
            target_role = resume_data.get('target_role', '지원 직무')
            summary = resume_data.get('resume_content', {}).get('evaluation', {}).get('summary', '')
            
            company_names = [
                f"{i+1}. {rec['metadata']['company_name']} ({rec['metadata']['job_title']})" 
                for i, rec in enumerate(recommendations)
            ]
            company_text = "\n".join(company_names)

            prompt = f"""
            당신은 커리어 컨설턴트입니다. 아래 지원자의 정보를 바탕으로 추천 기업에 대한 분석 리포트를 작성해주세요.

            [지원자 정보]
            - 희망 직무: {target_role}
            - 이력서 요약: {summary}

            [추천 기업 TOP 3]
            {company_text}

            [작성 요청사항]
            1. 지원자의 강점이 추천 기업들과 얼마나 잘 맞는지 3~4문장으로 분석해주세요.
            2. 3번째 기업은 다른 기업들과 어떤 점에서 차별화된 기회(유연한 추천)인지 언급해주세요.
            3. 격려의 메시지로 마무리해주세요.
            4. 말투는 정중하고 전문적인 톤(해요체)으로 작성해주세요.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",  # 또는 gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "전문적인 커리어 컨설턴트입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ AI 리포트 생성 실패: {e}")
            return "AI 리포트를 생성하는 도중 오류가 발생했습니다."

# 싱글톤 인스턴스
matching_engine = MatchingEngine()
