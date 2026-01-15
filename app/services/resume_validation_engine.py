import json
import re
import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# 필수 라이브러리 로드
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
except ImportError as e:
    print(f"⚠️ 필수 라이브러리가 설치되지 않았습니다: {e}")
    # 실제 환경에서는 로그를 남기거나 에러를 raise 할 수 있음

# ==========================================
# 1. 데이터 로드 및 경로 설정
# ==========================================
def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리를 반환합니다.
    app/services/resume_validation_engine.py -> 프로젝트 루트
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

def get_data_path() -> Path:
    """
    데이터 디렉토리 경로를 반환합니다.
    """
    return get_project_root() / "app" / "data"

class DataLoader:
    """
    기존 데이터 로더 유지 (필요 시 다른 메타데이터 접근용)
    """
    def __init__(self):
        self.base_path = get_data_path()
        self.file_names = {
            "resumes": "final_resume_600.json",
            "companies": "company_50_pool.json",
            "metadata": "final_metadata_600.json"
        }
        self.data = {}
        # 필요하다면 여기서 파일을 로드할 수 있음. 
        # 현재 MatchingEngine은 pickle 파일을 직접 로드하므로,
        # 여기서는 경로 확인 정도만 수행하거나 비워둘 수 있음.
        
    def normalize(self, text):
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())

# ==========================================
# 2. 매칭 엔진 (Hybrid Vector + Keyword)
# ==========================================
class MatchingEngine:
    """
    [Core Engine]
    하이브리드 매칭 (벡터 55% + 키워드 35%) + [Metadata Bonus 10%]
    + [Smart Calibration] (랜덤이 아닌, 실력 기반 점수 매핑)
    """

    # 1. 매칭 가중치 (Total 1.0)
    WEIGHT_VECTOR = 0.55
    WEIGHT_KEYWORD = 0.35
    BONUS_ROLE_MATCH = 0.10

    # 2. 등급별 추천 기업 티어 (Quota)
    TIER_RULES = {
        "S": ["Top", "Top", "Mid"],
        "A": ["Top", "Mid", "Mid"],
        "B": ["Mid", "Mid", "Low"],
        "C": ["Mid", "Low", "Low"],
        "F": ["Low", "Low", "Low"]
    }

    # 3. 목표 점수 구간 (사용자 만족용)
    SCORE_RANGES = [
        (88.0, 97.0), # Rank 1
        (77.0, 86.0), # Rank 2
        (66.0, 75.0)  # Rank 3
    ]

    # [NEW] 현실적인 Raw Score 기준점 (정규화 후 기준)
    RAW_SCORE_MIN = 0.30
    RAW_SCORE_MAX = 0.95
    GAP_THRESHOLD = 0.50  # 0.5점(50점) 미만이면 경고

    def __init__(self):
        print("🚀 매칭 엔진(Matching Engine) 초기화 중...")
        
        self.base_path = get_data_path()
        self.model_name = "jhgan/ko-sroberta-multitask"
        
        # OpenAI Client (환경변수에서 키 로드)
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

        # Model Load
        print(f"   -> 모델 로드 중: {self.model_name}")
        # 모델 로딩은 시간이 걸릴 수 있으므로, 실제 운영 환경에서는 싱글톤 패턴이나 시작 시 로드를 고려해야 함
        # 여기서는 인스턴스 생성 시 로드
        self.model = SentenceTransformer(self.model_name)

        # Data Load
        self.company_data = self._load_company_vectors()
        
        # DataLoader 인스턴스 (보조용)
        self.dl = DataLoader()

    def _load_company_vectors(self):
        """pkl 파일 로드"""
        pkl_path = self.base_path / "company_jd_vectors.pkl"
        
        if not pkl_path.exists():
            print(f"❌ 기업 벡터 파일을 찾을 수 없습니다: {pkl_path}")
            return None
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'vectors' in data:
                print(f"   -> 기업 데이터 로드 완료: {len(data['companies'])}개 기업")
                return data
            else:
                print("❌ pkl 파일 구조 오류")
                return None
        except Exception as e:
            print(f"❌ 데이터 로딩 오류: {e}")
            return None

    def _calculate_keyword_score(self, resume_text: str, tech_stack: List[str]) -> float:
        if not tech_stack: return 0.5
        resume_lower = resume_text.lower()
        match_count = sum(1 for t in tech_stack if t.lower() in resume_lower)
        return match_count / len(tech_stack)

    def _calculate_metadata_bonus(self, candidate_role: str, company_target_roles: List[str]) -> float:
        if not candidate_role or not company_target_roles: return 0.0
        cand_role_lower = candidate_role.lower()
        for role in company_target_roles:
            role_lower = role.lower()
            if role_lower in cand_role_lower or cand_role_lower in role_lower:
                return self.BONUS_ROLE_MATCH
            if "fullstack" in cand_role_lower and role_lower in ["backend", "frontend"]:
                return self.BONUS_ROLE_MATCH * 0.8
            if "ai" in cand_role_lower and role_lower in ["nlp", "llm", "vision"]:
                return self.BONUS_ROLE_MATCH
        return 0.0

    def _normalize_vector_score(self, val: float) -> float:
        """
        [New] S-BERT Cosine Similarity 정규화
        기계적 유사도(0.15~0.75)를 인간이 이해하는 점수(0.0~1.0)로 변환
        """
        min_bound = 0.15
        max_bound = 0.75

        normalized = (val - min_bound) / (max_bound - min_bound)
        return max(0.0, min(1.0, normalized))

    def _map_score_to_range(self, raw_score: float, target_min: float, target_max: float) -> float:
        """
        [Dynamic Scaling] 현실적인 입력 범위(Raw Score)를 목표 범위로 매핑
        """
        input_min, input_max = self.RAW_SCORE_MIN, self.RAW_SCORE_MAX

        normalized = (raw_score - input_min) / (input_max - input_min)
        normalized = max(0.0, min(1.0, normalized))

        scaled_score = target_min + (normalized * (target_max - target_min))
        return round(scaled_score, 1)

    def _categorize_companies(self, all_companies, vector_scores, resume_text, candidate_role):
        buckets = {"Top": [], "Mid": [], "Low": []}

        for idx, comp in enumerate(all_companies):
            # 1. 벡터 점수 정규화
            v_raw = float(vector_scores[idx])
            v_norm = self._normalize_vector_score(v_raw)

            # 2. 키워드 점수
            k_score = self._calculate_keyword_score(resume_text, comp.get('tech_stack', []))

            # Semantic Rescue (키워드 0점 구제)
            if k_score == 0.0 and v_norm > 0.5: k_score = 0.2

            # 3. 가중치 적용 합산
            hybrid_score = (v_norm * self.WEIGHT_VECTOR) + (k_score * self.WEIGHT_KEYWORD)

            # 4. 메타데이터 보너스
            meta_bonus = self._calculate_metadata_bonus(candidate_role, comp.get('target_roles', []))

            final_raw_score = hybrid_score + meta_bonus
            final_raw_score = min(1.0, final_raw_score)

            comp_data = {
                "metadata": {
                    "company_name": comp["name"],
                    "job_title": ", ".join(comp.get("target_roles", [])),
                    "industry": comp["industry"],
                    "tier": comp.get("tier", "Low")
                },
                "tech_stack": comp.get("tech_stack", []), # 내부 로직용
                "raw_score": final_raw_score,
                "vector_raw": round(v_raw, 2),
                "vector_norm": round(v_norm, 2),
                "keyword_raw": round(k_score, 2),
                "meta_bonus": round(meta_bonus, 2)
            }
            # MatchResult 모델 호환성을 위해 flat 하게 저장하지 않고 metadata 구조 유지하되,
            # 내부 로직에서는 comp_data 접근
            
            # API 호환을 위해 company_name 등은 metadata 안에 넣고, 
            # 추천 로직 내에서는 편의상 키 접근

            tier = comp.get("tier", "Low")
            if tier not in buckets: tier = "Low"
            buckets[tier].append(comp_data)

        for t in buckets:
            buckets[t].sort(key=lambda x: x['raw_score'], reverse=True)

        return buckets

    def _convert_resume_to_text(self, resume_input: dict) -> str:
        """
        이력서 JSON 객체를 임베딩 가능한 텍스트로 변환
        """
        parts = []
        
        # 1. 스킬
        content = resume_input.get('resume_content', {})
        skills = content.get('skills', {})
        essential = skills.get('essential', [])
        additional = skills.get('additional', [])
        all_skills = essential + additional
        if all_skills:
            parts.append(f"Technical Skills: {', '.join(all_skills)}")
            
        # 2. 경력 (Key Tasks 위주)
        experiences = content.get('professional_experience', [])
        for exp in experiences:
            role = exp.get('role', '')
            tasks = exp.get('key_tasks', [])
            parts.append(f"Role: {role}")
            if tasks:
                parts.append(f"Tasks: {', '.join(tasks)}")
                
        # 3. 프로젝트
        projects = content.get('project_experience', [])
        for proj in projects:
            title = proj.get('project_title', '')
            achievements = proj.get('key_achievements', [])
            parts.append(f"Project: {title}")
            if achievements:
                parts.append(f"Achievements: {', '.join(achievements)}")
                
        # 4. 분석된 직무 (Target Role)
        classification = resume_input.get('classification', {})
        role = classification.get('predicted_role', '')
        if not role:
             role = resume_input.get('target_role', '')
        if role:
            parts.append(f"Target Role: {role}")
            
        return "\n".join(parts)

    def generate_xai_feedback(self, resume_input: dict, recommendations: List[Dict]) -> str:
        """
        [기능 수정] 전체 추천 기업 목록을 기반으로
        친근하고 구체적인 'AI 코치' 스타일의 종합 피드백 문장 생성
        """
        feedback_lines = ["\n종합 AI 코치 의견:"]

        if not recommendations:
            feedback_lines.append("제공된 이력서에 맞는 추천 기업을 찾지 못했습니다. 이력서 내용을 점검해주세요.")
            return "\n".join(feedback_lines)

        # 1. 최고 매칭 기업 정보 분석
        top_rec = recommendations[0]
        # comp_data 구조가 _categorize_companies에서 정의됨
        # metadata 내부에 company_name이 있음
        top_company_name = top_rec['metadata']['company_name']
        top_score = top_rec['match_score'] # recommend_companies에서 계산되어 추가됨
        top_note = top_rec.get('note', '')

        # 이력서의 핵심 스킬 추출
        content = resume_input.get('resume_content', {})
        skills = content.get('skills', {})
        resume_essential_skills = set(skills.get('essential', []))
        resume_additional_skills = set(skills.get('additional', []))
        resume_all_skills = resume_essential_skills.union(resume_additional_skills)

        exp_tasks_summary = []
        for exp in content.get('professional_experience', [])[:1]: # Top 1 experience
            exp_tasks_summary.extend(exp.get('key_tasks', [])[:2]) # Top 2 tasks
        exp_summary_str = ", ".join(exp_tasks_summary) if exp_tasks_summary else "다양한 프로젝트 경험"

        # 2. 종합 평가 멘트
        feedback_lines.append(f"이력서를 종합적으로 분석해 보니, 최고 매칭 기업인 **{top_company_name}**에서 {top_score}점으로 '{top_note}' 평가를 받았습니다.")

        if top_score >= 88:
            skills_str = ', '.join(list(resume_all_skills)[:3])
            feedback_lines.append(f"지원자님의 **{skills_str}** 등의 핵심 역량과 **{exp_summary_str}** 경험이 해당 기업의 요구사항과 매우 잘 맞아 떨어집니다. 이 강점을 적극적으로 어필하면 좋은 결과가 있을 것입니다! 🚀")
        elif top_score >= 76:
            skills_str = ', '.join(list(resume_all_skills)[:2])
            feedback_lines.append(f"전반적으로 안정적인 기술 핏을 보여주며, 특히 **{skills_str}** 역량은 충분합니다. 면접에서 **{exp_summary_str}** 경험과 성장 가능성을 효과적으로 전달한다면 합격권에 들 수 있습니다! 💪")
        else:
            # 전체 추천 목록에서 'Skill Gap'이 있는 회사들을 찾아 부족한 스킬셋을 언급
            all_missing_skills = set()
            for rec in recommendations:
                comp_stack = set(rec.get('tech_stack', []))
                missing = comp_stack - resume_all_skills
                if missing: all_missing_skills.update(list(missing)[:1])
            
            missing_str = ", ".join(list(all_missing_skills)[:3]) if all_missing_skills else "특정 기술 스택"

            feedback_lines.append(f"아쉽게도 추천된 기업들, 특히 **{top_company_name}**에서는 **{missing_str}** 관련 역량에 대한 보완이 필요하다는 의견이 있었습니다.")
            feedback_lines.append("이력서에서 언급된 부족 스킬에 대한 학습 계획이나 관련 프로젝트 경험을 강조하여 성장 가능성을 보여주는 것이 중요합니다. 포기하지 않고 꾸준히 발전하는 모습을 보여주세요! 🌟")

        return "\n".join(feedback_lines)

    def recommend(self, resume_input: dict):
        """
        FastAPI 라우터 호환용 메인 메소드
        """
        # [방어 코드] 기업 데이터 확인
        if not self.company_data:
            return [], "시스템 에러: 기업 데이터(Vector DB)가 로드되지 않았습니다."

        # 1. 이력서 텍스트 변환
        resume_text = self._convert_resume_to_text(resume_input)
        
        # 2. 직무 파악
        classification = resume_input.get('classification', {})
        role = classification.get('predicted_role', '')
        if not role:
             role = resume_input.get('target_role', 'backend') # default

        # 3. 벡터 임베딩 및 유사도 계산
        query_vector = self.model.encode([resume_text])
        all_vectors = self.company_data['vectors']
        vector_scores = cosine_similarity(query_vector, all_vectors)[0]

        # 4. 버킷팅 및 점수 계산
        buckets = self._categorize_companies(self.company_data['companies'], vector_scores, resume_text, role)

        # 5. 등급 기반 기업 선정 (candidate_grade는 DTO에 없으므로 기본 B로 가정하거나 점수 기반 역산 가능하나, 여기선 B default)
        # resume_input에 grade 정보가 있다면 사용
        # resume_evaluation = resume_input.get('resume_evaluation', {})
        # candidate_grade = resume_evaluation.get('grade', 'B') if resume_evaluation else 'B'
        candidate_grade = "B" # 기본값

        target_slots = self.TIER_RULES.get(candidate_grade, self.TIER_RULES["B"])
        final_selection = []
        used_companies = set()

        for required_tier in target_slots:
            selected = None
            for comp in buckets.get(required_tier, []):
                comp_name = comp['metadata']['company_name']
                if comp_name not in used_companies:
                    selected = comp
                    break

            if not selected:
                search_order = ["Top", "Mid", "Low"] if required_tier == "Top" else ["Mid", "Low", "Top"]
                for tier in search_order:
                    for comp in buckets.get(tier, []):
                        comp_name = comp['metadata']['company_name']
                        if comp_name not in used_companies:
                            selected = comp
                            break
                    if selected: break

            if selected:
                used_companies.add(selected['metadata']["company_name"])
                final_selection.append(selected)

        # 6. 점수 매핑 (Smart Calibration) 및 결과 포맷팅
        formatted_results = []
        for i, res in enumerate(final_selection):
            # Dynamic Scaling
            if i < len(self.SCORE_RANGES):
                min_s, max_s = self.SCORE_RANGES[i]
                final_score = self._map_score_to_range(res['raw_score'], min_s, max_s)
            else:
                final_score = round(res['raw_score'] * 100, 1)

            # Note 설정
            if i == 0:
                note = "🏆 Best Match"
            elif res['raw_score'] < self.GAP_THRESHOLD:
                note = "⚠️ Skill Gap"
            else:
                note = "✅ High Fit"

            # 내부 딕셔너리 업데이트 (feedback 생성용)
            res['match_score'] = final_score
            res['note'] = note
            
            # API 반환용 구조로 변환
            # MatchResult: company_name, match_score, tier, match_type, reason
            # api/routes.py 호환을 위해 raw_score, is_exact_match 추가
            formatted_results.append({
                "metadata": res['metadata'], # 기존 구조 호환
                "company_name": res['metadata']['company_name'], # API 필드
                "match_score": final_score,
                "tier": res['metadata']['tier'],
                "match_type": note,
                "reason": f"Tech Match: {res['keyword_raw']*100:.0f}%, Vector: {res['vector_norm']:.2f}",
                
                # [Legacy Support] api/routes.py 호환
                "raw_score": final_score, 
                "is_exact_match": (note == "🏆 Best Match") or (final_score >= 85),
                
                # 내부 로직용 필드 유지 (feedback 용)
                "tech_stack": res['tech_stack'],
                "note": note
            })

        # 7. 피드백 생성
        report = self.generate_xai_feedback(resume_input, formatted_results)

        return formatted_results, report

# Singleton Instance
resume_validation_engine = MatchingEngine()
