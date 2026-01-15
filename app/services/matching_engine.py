import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
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
    print("⚠️ [Warning] 필수 라이브러리(sentence_transformers, sklearn, openai)가 없습니다.")

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
    
    def __init__(self, base_path: Optional[str] = None, openai_api_key: Optional[str] = None):
        print("🚀 매칭 엔진(Matching Engine) 초기화 중...")
        
        # 경로 자동 설정 (app/services/matching_engine.py -> project_root/data)
        if base_path is None:
            current_dir = Path(__file__).resolve().parent
            self.base_path = current_dir.parent.parent / "data"
        else:
            self.base_path = Path(base_path)

        # 모든 속성 사전 초기화
        self.model = None
        self.company_vectors = None
        self.company_metadata = []
        self.client = None
        
        # 1. 라이브러리 체크
        if not SentenceTransformer:
            print("⚠️ [Critical] sentence_transformers 라이브러리가 없습니다.")
            return

        # 2. OpenAI 클라이언트 설정
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        if not self.client:
            print("⚠️ [Warning] OpenAI API Key가 없습니다. AI 리포트 기능이 제한됩니다.")
        
        # 3. 데이터 및 모델 로드
        try:
            print(f"📂 데이터 로딩 중... ({self.base_path})")
            
            # (1) 벡터 데이터 로드 (.npy)
            vector_path = self.base_path / "final_embedded_dataset_600.npy"
            if vector_path.exists():
                self.company_vectors = np.load(vector_path)
                print(f"  - 벡터 데이터 로드 완료: {self.company_vectors.shape}")
            else:
                print(f"❌ 벡터 파일을 찾을 수 없습니다: {vector_path}")
                print("👉 'final_embedded_dataset_600.npy' 파일이 data 폴더에 있는지 확인하세요.")
                self.company_vectors = None

            # (2) 메타데이터 로드 (.json)
            # engine용 메타데이터가 따로 있다면 그걸 쓰고, 없다면 pool 사용
            meta_path = self.base_path / "final_metadata_600.json"
            # 만약 final_metadata.json이 없으면 company_50_pool.json을 대체 사용 시도
            if not meta_path.exists():
                meta_path = self.base_path / "company_50_pool.json"

            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.company_metadata = json.load(f)
                print(f"  - 메타데이터 로드 완료: {len(self.company_metadata)}개")
            else:
                print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {meta_path}")
                self.company_metadata = []

            # (3) 임베딩 모델 로드
            print("🧠 임베딩 모델 로딩 중 (jhgan/ko-sroberta-multitask)...")
            # 로컬 캐시가 있으면 빠르지만, 처음엔 다운로드 시간 소요됨
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("✅ 모델 로드 완료")

        except Exception as e:
            print(f"❌ 초기화 중 에러 발생: {e}")
            self.model = None
            self.company_vectors = None
            self.company_metadata = []

    def recommend(self, resume_data: Dict[str, Any]) -> Tuple[List[Dict], str]:
        """
        이력서 데이터를 받아 추천 기업 리스트와 AI 리포트를 반환
        규칙: 1, 2위는 Target Role 일치(Exact), 3위는 유연한 추천(Flexible)
        """
        if not self.model or self.company_vectors is None or not self.company_metadata:
            return [], "⚠️ 매칭 엔진이 정상적으로 초기화되지 않았습니다 (데이터 누락 등)."

        # 1. 이력서 정보 추출
        resume_content = resume_data.get('resume_content', {})
        target_role = resume_data.get('target_role', '').lower()
        
        # Pydantic 모델에서 dict로 변환되어 들어올 때 필드 접근 처리
        if hasattr(resume_content, 'dict'): resume_content = resume_content.dict()
        
        eval_data = resume_data.get('resume_evaluation', {}) or {}
        if hasattr(eval_data, 'dict'): eval_data = eval_data.dict()

        # classification 정보가 없으면 임시 생성 (Resume 스키마에 classification이 없으므로)
        classification = {
            'keywords': resume_content.get('skills', {}).get('essential', []) + 
                        resume_content.get('skills', {}).get('additional', []),
            'predicted_role': target_role
        }
        
        resume_keywords = set([k.lower() for k in classification.get('keywords', [])])
        
        # 2. 개선된 쿼리 텍스트 생성
        query_text = self._build_query_text(target_role, classification, eval_data)
        query_vector = self.model.encode(query_text)

        # 3. 코사인 유사도 계산
        # 벡터 크기 불일치 방지
        if query_vector.shape[0] != self.company_vectors.shape[1]:
             print(f"⚠️ 벡터 차원 불일치! Query: {query_vector.shape}, DB: {self.company_vectors.shape}")
             return [], "임베딩 모델 버전이 데이터 생성 시점과 다릅니다."

        raw_scores = cosine_similarity([query_vector], self.company_vectors)[0]

        # 4. Top N 후보군 추출 (넉넉하게 30개)
        top_n_indices = np.argsort(raw_scores)[::-1][:30]
        
        # 5. 하이브리드 스코어링 적용
        candidates = []
        for idx in top_n_indices:
            # 인덱스 범위 체크
            if idx >= len(self.company_metadata): continue
                
            company = self.company_metadata[idx]
            cosine_score = float(raw_scores[idx])
            
            # 직무 일치 여부 확인
            # company_pool.json에는 job_title이 없을 수 있음 -> target_roles로 대체
            if 'job_title' in company:
                job_title = company.get('job_title', '').lower()
            else:
                # target_roles 리스트를 문자열로 합쳐서 비교
                job_title = " ".join(company.get('target_roles', [])).lower()

            is_exact_match = self._check_exact_match(target_role, job_title)
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
        parts = []
        if target_role:
            parts.append(target_role)
            parts.append(target_role)
        
        predicted_role = classification.get('predicted_role', '')
        if predicted_role: parts.append(predicted_role)
        
        keywords = classification.get('keywords', [])
        if keywords:
            parts.extend(keywords)
            parts.extend(keywords)
        
        # eval_data 구조 대응 (ResumeEvaluation 객체일 수 있음)
        if isinstance(eval_data, dict):
             # summary가 문자열이거나 dict일 수 있음
             summary = eval_data.get('reasoning') or eval_data.get('summary') or ''
        else:
             summary = ''

        if summary: parts.append(str(summary))
        
        return " ".join(parts)

    def _check_exact_match(self, target_role: str, job_title: str) -> bool:
        if not target_role: return False
        if target_role in job_title: return True
        pm_keywords = ['pm', 'po', '기획', 'product', '프로덕트']
        if any(kw in target_role for kw in pm_keywords):
            if any(kw in job_title for kw in pm_keywords): return True
        return False

    def _check_related_role(self, target_role: str, job_title: str) -> bool:
        if not target_role: return False
        target_group = None
        for group, keywords in RELATED_ROLES.items():
            if any(kw.lower() in target_role for kw in keywords):
                target_group = group
                break
        if not target_group: return False
        for kw in RELATED_ROLES[target_group]:
            if kw.lower() in job_title: return True
        return False

    def _calculate_hybrid_score(self, cosine_score: float, is_exact_match: bool, resume_keywords: set, company: Dict) -> float:
        base_score = cosine_score * 60
        role_bonus = 20 if is_exact_match else 0
        
        company_stack = company.get('tech_stack', [])
        company_skills = set(s.lower() for s in company_stack)
        
        if resume_keywords and company_skills:
            matching_count = len(resume_keywords & company_skills)
            total_keywords = len(resume_keywords)
            keyword_score = (matching_count / max(total_keywords, 1)) * 20
        else:
            keyword_score = 0
        
        return base_score + role_bonus + keyword_score

    def _normalize_scores(self, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return candidates
        max_score = max(c['raw_score'] for c in candidates)
        min_score = min(c['raw_score'] for c in candidates)
        
        if max_score == min_score:
            for c in candidates: c['raw_score'] = 85.0
            return candidates
        
        for c in candidates:
            normalized = 60 + (c['raw_score'] - min_score) / (max_score - min_score) * 35
            c['raw_score'] = round(normalized, 1)
        return candidates

    def _apply_filtering_rules(self, candidates: List[Dict], target_role: str = "") -> List[Dict]:
        exact_matches = [c for c in candidates if c['is_exact_match']]
        related_matches = [c for c in candidates if not c['is_exact_match'] and c.get('is_related_role', False)]
        other_matches = [c for c in candidates if not c['is_exact_match'] and not c.get('is_related_role', False)]
        
        flexible_matches = [c for c in related_matches if c['raw_score'] >= 65]
        flexible_matches.extend([c for c in other_matches if c['raw_score'] >= 65])
        
        if not flexible_matches:
            flexible_matches = related_matches + other_matches
        
        result = []
        # 1, 2위 Exact 우선
        for _ in range(2):
            if exact_matches: result.append(exact_matches.pop(0))
            elif flexible_matches: result.append(flexible_matches.pop(0))
        # 3위 Flexible 우선
        if flexible_matches: result.append(flexible_matches.pop(0))
        elif exact_matches: result.append(exact_matches.pop(0))
            
        return result[:3]

    def _generate_ai_report(self, resume_data: Dict, recommendations: List[Dict]) -> str:
        if not self.client:
            return "OpenAI API Key 미설정으로 AI 리포트를 생성할 수 없습니다."
        try:
            target_role = resume_data.get('target_role', '지원 직무')
            summary = resume_data.get('resume_evaluation', {}).get('reasoning', '이력서 요약 없음')
            
            company_names = []
            for i, rec in enumerate(recommendations):
                c_name = rec['metadata'].get('company_name') or rec['metadata'].get('name')
                c_job = rec['metadata'].get('job_title') or " ".join(rec['metadata'].get('target_roles', []))
                company_names.append(f"{i+1}. {c_name} ({c_job})")
            
            company_text = "\n".join(company_names)
            prompt = f"""
            [지원자] 직무: {target_role}, 요약: {summary}
            [추천기업] {company_text}
            위 지원자에게 추천 기업이 왜 적합한지, 특히 3번째 기업은 어떤 차별점이 있는지 3문장으로 격려하며 요약해줘.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"AI 리포트 생성 오류: {e}"