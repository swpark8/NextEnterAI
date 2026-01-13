import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
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

# 연관 직무 그룹 정의 (Flexible 추천용)
RELATED_ROLES = {
    "기획": ["서비스 기획", "PO", "PM", "프로덕트", "마케팅", "운영", "전략"],
    "개발": ["Developer", "개발자", "엔지니어", "Engineer", "Fullstack", "Backend", "Frontend"],
    "데이터": ["데이터", "Data", "ML", "AI", "분석", "Analyst"],
    "디자인": ["디자인", "Design", "UX", "UI", "브랜드"],
}

class MatchingEngine:
    """
    [Hybrid RAG Engine]
    벡터 검색(S-BERT) + 하이브리드 스코어링 + AI 리포트(GPT)
    """
    
    def __init__(self, base_path: str = "./data", openai_api_key: Optional[str] = None):
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

        # 1. 이력서 정보 추출
        resume_content = resume_data.get('resume_content', {})
        target_role = resume_data.get('target_role', '').lower()
        
        eval_data = resume_content.get('evaluation', {})
        classification = resume_content.get('classification', {})
        
        # 이력서에서 키워드 추출
        resume_keywords = set([k.lower() for k in classification.get('keywords', [])])
        
        # 2. 개선된 쿼리 텍스트 생성
        query_text = self._build_query_text(target_role, classification, eval_data)
        query_vector = self.model.encode(query_text)

        # 3. 코사인 유사도 계산
        raw_scores = cosine_similarity([query_vector], self.company_vectors)[0]

        # 4. Top N 후보군 추출 (넉넉하게 30개)
        top_n_indices = np.argsort(raw_scores)[::-1][:30]
        
        # 5. 하이브리드 스코어링 적용
        candidates = []
        for idx in top_n_indices:
            company = self.company_metadata[idx]
            cosine_score = float(raw_scores[idx])
            
            # 직무 일치 여부 확인
            job_title = company.get('job_title', '').lower()
            is_exact_match = self._check_exact_match(target_role, job_title)
            
            # 연관 직무 여부 확인 (Flexible 추천용)
            is_related_role = self._check_related_role(target_role, job_title)
            
            # 하이브리드 점수 계산
            hybrid_score = self._calculate_hybrid_score(
                cosine_score=cosine_score,
                is_exact_match=is_exact_match,
                resume_keywords=resume_keywords,
                company=company
            )
            
            candidates.append({
                "metadata": company,
                "raw_score": hybrid_score,
                "cosine_score": cosine_score * 100,
                "is_exact_match": is_exact_match,
                "is_related_role": is_related_role
            })

        # 6. 점수 정규화 (상위 후보 기준)
        candidates = self._normalize_scores(candidates)

        # 7. 필터링 로직 (1,2위 Exact / 3위 Flexible)
        final_recommendations = self._apply_filtering_rules(candidates, target_role)

        # 8. AI 리포트 생성
        ai_report = self._generate_ai_report(resume_data, final_recommendations)
        
        return final_recommendations, ai_report

    def _build_query_text(self, target_role: str, classification: Dict, eval_data: Dict) -> str:
        """
        이력서 정보를 풍부하게 반영한 쿼리 텍스트 생성
        """
        parts = []
        
        # 1. 희망 직무 (가중치 높임 - 2번 반복)
        if target_role:
            parts.append(target_role)
            parts.append(target_role)
        
        # 2. 예측된 직무
        predicted_role = classification.get('predicted_role', '')
        if predicted_role:
            parts.append(predicted_role)
        
        # 3. 핵심 키워드 (2번 반복으로 가중치)
        keywords = classification.get('keywords', [])
        if keywords:
            parts.extend(keywords)
            parts.extend(keywords)
        
        # 4. 강점 (pros)
        pros = eval_data.get('pros', [])
        if pros:
            parts.extend(pros)
        
        # 5. 요약
        summary = eval_data.get('summary', '')
        if summary:
            parts.append(summary)
        
        return " ".join(parts)

    def _check_exact_match(self, target_role: str, job_title: str) -> bool:
        """
        직무 일치 여부 확인 (더 유연한 매칭)
        """
        if not target_role:
            return False
        
        # 직접 포함 확인
        if target_role in job_title:
            return True
        
        # PM/PO 특수 케이스
        pm_keywords = ['pm', 'po', '기획', 'product', '프로덕트']
        if any(kw in target_role for kw in pm_keywords):
            if any(kw in job_title for kw in pm_keywords):
                return True
        
        return False

    def _check_related_role(self, target_role: str, job_title: str) -> bool:
        """
        연관 직무 여부 확인 (Flexible 추천용)
        """
        if not target_role:
            return False
        
        # target_role이 속한 그룹 찾기
        target_group = None
        for group, keywords in RELATED_ROLES.items():
            if any(kw.lower() in target_role for kw in keywords):
                target_group = group
                break
        
        if not target_group:
            return False
        
        # job_title이 같은 그룹에 속하는지 확인
        for kw in RELATED_ROLES[target_group]:
            if kw.lower() in job_title:
                return True
        
        return False

    def _calculate_hybrid_score(
        self, 
        cosine_score: float, 
        is_exact_match: bool,
        resume_keywords: set,
        company: Dict
    ) -> float:
        """
        하이브리드 점수 계산
        - 코사인 유사도: 60%
        - 직무 일치 보너스: 20%
        - 키워드 매칭: 20%
        """
        # 1. 코사인 유사도 (60%)
        base_score = cosine_score * 60
        
        # 2. 직무 일치 보너스 (20%)
        role_bonus = 20 if is_exact_match else 0
        
        # 3. 키워드 매칭 점수 (20%)
        company_skills = set()
        for skill in company.get('tech_stack', []):
            company_skills.add(skill.lower())
        for skill in company.get('req_skills', []):
            company_skills.add(skill.lower())
        
        if resume_keywords and company_skills:
            matching_count = len(resume_keywords & company_skills)
            total_keywords = len(resume_keywords)
            keyword_score = (matching_count / max(total_keywords, 1)) * 20
        else:
            keyword_score = 0
        
        return base_score + role_bonus + keyword_score

    def _normalize_scores(self, candidates: List[Dict]) -> List[Dict]:
        """
        점수 정규화: 최고점을 95~100점 범위로 보정
        """
        if not candidates:
            return candidates
        
        max_score = max(c['raw_score'] for c in candidates)
        min_score = min(c['raw_score'] for c in candidates)
        
        if max_score == min_score:
            for c in candidates:
                c['raw_score'] = 85.0
            return candidates
        
        # 정규화: 최고점 -> 95, 최저점 -> 60 범위로 매핑
        for c in candidates:
            normalized = 60 + (c['raw_score'] - min_score) / (max_score - min_score) * 35
            c['raw_score'] = round(normalized, 1)
        
        return candidates

    def _apply_filtering_rules(self, candidates: List[Dict], target_role: str = "") -> List[Dict]:
        """
        요구사항: 1, 2위는 Exact Match, 3위는 Flexible Match (연관 직무 우선)
        """
        exact_matches = [c for c in candidates if c['is_exact_match']]
        
        # Flexible: 연관 직무 우선, 그 다음 기타
        related_matches = [c for c in candidates if not c['is_exact_match'] and c.get('is_related_role', False)]
        other_matches = [c for c in candidates if not c['is_exact_match'] and not c.get('is_related_role', False)]
        
        # Flexible 후보: 연관 직무 먼저, 그 다음 기타 (단, 최소 점수 65점 이상)
        flexible_matches = [c for c in related_matches if c['raw_score'] >= 65]
        flexible_matches.extend([c for c in other_matches if c['raw_score'] >= 65])
        
        # 만약 65점 이상이 없으면 그냥 연관 직무 사용
        if not flexible_matches:
            flexible_matches = related_matches + other_matches
        
        result = []
        
        # 1위, 2위 선정 (Exact Match 우선)
        for _ in range(2):
            if exact_matches:
                result.append(exact_matches.pop(0))
            elif flexible_matches:
                result.append(flexible_matches.pop(0))
                
        # 3위 선정 (Flexible Match 우선 - 연관 직무)
        if flexible_matches:
            result.append(flexible_matches.pop(0))
        elif exact_matches:
            result.append(exact_matches.pop(0))
            
        return result[:3]

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
            return response.choices[0].message.content or ""

        except Exception as e:
            print(f"❌ AI 리포트 생성 실패: {e}")
            return "AI 리포트를 생성하는 도중 오류가 발생했습니다."

# 싱글톤 인스턴스
matching_engine = MatchingEngine()
