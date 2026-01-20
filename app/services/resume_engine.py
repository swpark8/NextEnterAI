import json
import re
import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 필수 라이브러리 로드
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
except ImportError as e:
    print(f"[Warning] Required library not installed: {e}")
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
        print("[Start] Matching Engine initializing...")
        
        self.base_path = get_data_path()
        self.model_name = "jhgan/ko-sroberta-multitask"
        
        # OpenAI Client (환경변수에서 키 로드)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("[Warning] OPENAI_API_KEY is not set in .env file.")
        else:
            print(f"[Matching Engine] OpenAI API Key loaded successfully")
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
            print(f"[Error] Company vector file not found: {pkl_path}")
            return None
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'vectors' in data:
                print(f"   -> 기업 데이터 로드 완료: {len(data['companies'])}개 기업")
                return data
            else:
                print("[Error] pkl file structure error")
                return None
        except Exception as e:
            print(f"[Error] Data loading error: {e}")
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
            # 1. 완전 일치 또는 포함 관계
            if role_lower in cand_role_lower or cand_role_lower in role_lower:
                return self.BONUS_ROLE_MATCH
            
            # 2. Fullstack 유연한 매칭 (FE/BE 모두 인정)
            if "fullstack" in cand_role_lower and (role_lower in ["backend", "frontend"]):
                return self.BONUS_ROLE_MATCH * 0.8
            
            # 3. AI/LLM Engineer 특화 매칭
            if "ai" in cand_role_lower or "llm" in cand_role_lower:
                if any(kw in role_lower for kw in ["nlp", "llm", "vision", "ai", "ml", "data"]):
                    return self.BONUS_ROLE_MATCH
            
            # 4. UI/UX Designer 특화 매칭
            if "ui/ux" in cand_role_lower or "designer" in cand_role_lower:
                if any(kw in role_lower for kw in ["design", "ui", "ux", "product", "creative"]):
                    return self.BONUS_ROLE_MATCH
                    
        return 0.0

    def _check_role_relevance(self, target_category: str, current_role: str) -> bool:
        """
        [Team Rule] 지원 직무와 이전 경력 직무 간의 유사성 판단
        """
        if not target_category or not current_role: return False
        target_category = target_category.lower()
        current_role = current_role.lower()
        
        # 기본 키워드 정의
        relevance_keywords = [target_category, '기획', '개발', 'developer', 'manager', 'engineer', 'design']
        
        # 신규 직무별 확장 키워드
        if "designer" in target_category or "ui/ux" in target_category:
            relevance_keywords.extend(['art', 'creative', 'ux', 'ui', '디자인', '디자이너', '퍼블리셔'])
        
        if "ai" in target_category or "llm" in target_category:
            relevance_keywords.extend(['researcher', 'scientist', 'nlp', 'ml', 'data', 'lab', '연구원'])
        
        if any(kw in current_role for kw in relevance_keywords):
            return True
        return False

    def _normalize_vector_score(self, val: float) -> float:
        """
        [New] S-BERT Cosine Similarity 정규화
        기계적 유사도(0.15~0.75)를 인간이 이해하는 점수(0.0~1.0)로 변환
        """
        min_bound = 0.15
        max_bound = 0.75

        normalized = (val - min_bound) / (max_bound - min_bound)
        return max(0.0, min(1.0, normalized))

    def get_grade(self, score: float) -> str:
        """점수 기반 등급 판정"""
        if score >= 90: return "S"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        return "F"

    def verify_and_regrade(self, resume_input: dict, final_raw_score: float) -> float:
        """
        [Ghost F 해결] 텍스트 매칭은 준수하나 점수가 낮게 나온 경우 보정
        경력(40%), 프로젝트(30%), 기술(20%), 학력(10%) 가중치 기반 정밀 재채점
        """
        content = resume_input.get('resume_content', {})
        
        # 1. 학력 점수 (10%)
        edu_score = 0.0
        for edu in content.get('education', []):
            major = edu.get('major', '').lower()
            if any(kw in major for kw in ['컴퓨터', 'computer', '소프트웨어', 'software', 'IT', '전산']):
                edu_score = 0.1
                break
        
        # 2. 경력 점수 (40%)
        exp_score = 0.0
        experiences = content.get('professional_experience', [])
        if experiences:
            exp_score = 0.2 # 기본 경력 보유
            for exp in experiences:
                period = str(exp.get('period', ''))
                # 3년 이상 경력 시 가중치 최대
                if any(kw in period for kw in ['36개월', '3년', '48개월', '4년', '5년', '60개월']):
                    exp_score = 0.4
                    break
        
        # 3. 프로젝트 점수 (30%)
        proj_score = 0.0
        if content.get('project_experience', []):
            proj_score = 0.3

        # 4. 기술 점수 (기존 final_raw_score 활용 - 20%)
        tech_score = final_raw_score * 0.2

        # 최종 보정 점수
        refined_score = tech_score + exp_score + proj_score + edu_score
        
        # [Ghost F 방어] 학력과 경력이 양호하면 최소 C등급(0.6) 하한선 보장
        if (edu_score >= 0.1 or exp_score >= 0.3) and refined_score < 0.6:
            refined_score = 0.6
            
        return min(1.0, refined_score)

    def _map_score_to_range(self, raw_score: float, target_min: float, target_max: float) -> float:
        """
        [Dynamic Scaling] 현실적인 입력 범위(Raw Score)를 목표 범위로 매핑
        """
        input_min, input_max = self.RAW_SCORE_MIN, self.RAW_SCORE_MAX

        normalized = (raw_score - input_min) / (input_max - input_min)
        normalized = max(0.0, min(1.0, normalized))

        scaled_score = target_min + (normalized * (target_max - target_min))
        return round(scaled_score, 1)

    def _categorize_companies(self, all_companies, vector_scores, resume_input, candidate_role):
        buckets = {"Top": [], "Mid": [], "Low": []}
        resume_text = self._convert_resume_to_text(resume_input)
        
        # 0. 이력서 기본 정보 추출 (F등급 판정용)
        content = resume_input.get('resume_content', {})
        experiences = content.get('professional_experience', [])
        
        # 경력 월수 추출
        actual_months = 0
        current_exp_role = ""
        if experiences:
            period_str = str(experiences[0].get('period', '0'))
            current_exp_role = experiences[0].get('role', '')
            nums = re.findall(r'\d+', period_str)
            actual_months = int(nums[0]) if nums else 0
            
        # 직무 관련성 체크
        is_relevant_role = self._check_role_relevance(candidate_role, current_exp_role)

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

            base_hybrid_score = hybrid_score + meta_bonus
            
            # [추가] 직무 불일치 감점 (Conflict Penalty)
            # 메타데이터 보너스가 0인데 벡터 점수만 높은 경우, 실제 직무가 다를 확률이 높으므로 감점
            if meta_bonus == 0 and v_norm > 0.4:
                base_hybrid_score -= 0.15 # 약 15점 감점
            
            # 5. [Ghost F 해결] 정밀 재채점 (학력/경력 가중치 반영)
            final_raw_score = self.verify_and_regrade(resume_input, base_hybrid_score)

            # 6. [Team Rule] F등급 강제 판정 로직 (무경력/무관직무/기술매칭0)
            is_forced_f = False
            f_reason = ""
            if actual_months == 0:
                is_forced_f = True
                f_reason = "무경력자(신입)"
            elif not is_relevant_role:
                is_forced_f = True
                f_reason = "직무 불일치"
            elif k_score == 0 and v_norm < 0.3: # 기술 매칭이 매우 낮은 경우
                is_forced_f = True
                f_reason = "기술 역량 부족"

            if is_forced_f:
                # F등급 점수 제한 (최대 59점)
                final_raw_score = min(final_raw_score, 0.59)
            else:
                # C등급 이상 점수 보정 (최소 60점)
                if final_raw_score < 0.60:
                    final_raw_score = 0.60

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
                "meta_bonus": round(meta_bonus, 2),
                "is_forced_f": is_forced_f,
                "f_reason": f_reason
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
        [기능 강화] 전문적이고 객관적인 AI 피드백 생성
        - 등급별 톤 조절 (S/A/B: 긍정/전문, C/F: 냉철/분석)
        - 직무 적합도 상세 분석 및 보강 제안 포함
        """
        feedback_lines = ["\n종합 AI 코치 의견:"]

        if not recommendations:
            feedback_lines.append("분석 결과, 현재 이력서에 부합하는 추천 기업을 식별할 수 없습니다. 이력서의 기술 스택 및 경력 기술을 재점검하여 주시기 바랍니다.")
            return "\n".join(feedback_lines)

        # 1. 기본 정보 추출
        top_rec = recommendations[0]
        top_company_name = top_rec['metadata']['company_name']
        top_score = top_rec['match_score']
        top_note = top_rec.get('note', '')
        
        classification = resume_input.get('classification', {})
        predicted_role = classification.get('predicted_role') or resume_input.get('target_role', '미지정 직무')
        
        content = resume_input.get('resume_content', {})
        resume_all_skills = set(content.get('skills', {}).get('essential', [])).union(set(content.get('skills', {}).get('additional', [])))
        experiences = content.get('professional_experience', [])
        projects = content.get('project_experience', [])

        # 2. [기업 매칭 결과] 섹션
        feedback_lines.append(f"\n[기업 매칭 결과]")
        feedback_lines.append(f"대상 기업: {top_company_name}")
        feedback_lines.append(f"평가 등급: {top_note} ({top_score}점)")
        
        tone_summary = "긍정적" if top_score >= 76 else "분석적"
        feedback_lines.append(f"분석 요약: 해당 이력서는 {predicted_role} 포지션에 대해 {tone_summary}인 정합성을 보이고 있습니다.")

        # 3. [직무 적합도 상세 분석] 섹션
        feedback_lines.append(f"\n[직무 적합도 상세 분석]")
        
        # 기술 역량 분석
        tech_match_pct = int(top_rec.get('keyword_raw', 0) * 100)
        skills_list = list(resume_all_skills)
        skills_str = ', '.join(skills_list[:3]) if skills_list else "기초 역량"
        if tech_match_pct >= 80:
            tech_fit_msg = f"{skills_str} 중심의 핵심 역량이 기업 요구사항과 매우 높은 일치도를 보입니다."
        elif tech_match_pct >= 50:
            tech_fit_msg = f"{skills_str} 등 주요 기술 스택을 보유하고 있으나, 실무 활용 역량에 대한 보완이 권장됩니다."
        else:
            tech_fit_msg = "지원 직무에 필요한 핵심 기술 스택과 현재 보유하신 역량 간의 차이가 식별되었습니다."
        feedback_lines.append(f"- 기술 역량: {tech_fit_msg}")

        # 실무 경험 분석
        exp_count = len(experiences)
        if exp_count >= 1:
            exp_role = experiences[0].get('role', '관련 직무')
            exp_fit_msg = f"{exp_role} 경력을 통한 실무 기여 가능성이 높음으로 분석됩니다."
        else:
            exp_fit_msg = "실무 경력 증빙이 부족하여, 프로젝트 경험을 통한 역량 증명이 요구됩니다."
        feedback_lines.append(f"- 실무 경험: {exp_fit_msg}")

        # 직무 연관성 분석
        relevance_pct = int(top_rec.get('vector_norm', 0) * 100)
        feedback_lines.append(f"- 직무 연관성: 지원하신 직무와 보유하신 경력 간의 연관성은 {relevance_pct}% 수준입니다.")

        # 4. [AI 보강 제안] 섹션
        feedback_lines.append(f"\n[AI 보강 제안]")
        
        proposals = []
        if top_score >= 76: # S/A/B
            proj_title = projects[0].get('project_title', '주요 프로젝트') if projects else "수행 프로젝트"
            proposals.append(f"1. 핵심 성과 수치화: {proj_title} 경험의 성과를 정량적 지표(KPI)로 명시하여 객관성을 확보하십시오.")
            proposals.append(f"2. 직무 전문성 강조: 면접 시 {skills_str}를 활용한 문제 해결 사례를 구체적으로 어필하시기 바랍니다.")
        else: # C/F
            if not experiences:
                proposals.append("1. 직무 경력 보완: 지원 직무와 직접적으로 연관된 인턴십 또는 실무 프로젝트 경험을 확보하십시오.")
            else:
                proposals.append(f"1. 이력서 재구성: 현재의 {experiences[0].get('role', '이전 직무')} 중심 기술을 지원 직무인 {predicted_role} 관점으로 재해석하여 기술하십시오.")
            
            comp_stack = set(top_rec.get('tech_stack', []))
            missing = list(comp_stack - resume_all_skills)[:2]
            if missing:
                proposals.append(f"2. 기술 스택 확충: 부족한 {', '.join(missing)} 관련 역량을 학습하고 이를 활용한 포트폴리오를 추가하십시오.")
            else:
                proposals.append("2. 프로젝트 상세화: 수행하신 프로젝트의 기술적 난이도와 본인의 기여도를 더 구체적으로 기술하십시오.")

        feedback_lines.extend(proposals)

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
        buckets = self._categorize_companies(self.company_data['companies'], vector_scores, resume_input, role)

        # 5. 등급 기반 기업 선정 (Smart Regrading 반영)
        # resume_input에 이미 분석된 등급이 있다면 사용
        resume_evaluation = resume_input.get('evaluation') or {}
        candidate_grade = resume_evaluation.get('grade', 'B')
        
        # [Team Rule] 엔진이 판단한 강제 F 조건이 있는지 확인
        all_comp_data = []
        for t in buckets: all_comp_data.extend(buckets[t])
        
        is_candidate_forced_f = any(c.get('is_forced_f', False) for c in all_comp_data)
        
        if is_candidate_forced_f:
            candidate_grade = "F"
            print(f"   -> [Team Rule] Grade forced to F due to lack of experience or unrelated role.")
        elif candidate_grade == "F":
            # 분석 API에서는 F를 줬으나, 엔진 점수가 높게 나온 경우 (Ghost F 구제)
            # 전체 기업 평균 점수를 기반으로 잠재 등급 확인
            all_raw_scores = [c['raw_score'] for c in all_comp_data]
            if all_raw_scores:
                avg_score = sum(all_raw_scores) / len(all_raw_scores)
                if avg_score > 0.6: # C등급 이상 점수가 충분히 나옴
                    candidate_grade = "C"
                    print(f"   -> [Ghost F Recovery] Grade F -> {candidate_grade} (Avg Score: {avg_score:.2f})")

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

            # Note 설정 및 match_level 매핑
            if i == 0:
                note = "Best Match"
                match_level = "BEST"
            elif res['raw_score'] < self.GAP_THRESHOLD:
                note = "Skill Gap"
                match_level = "GAP"
            else:
                note = "High Fit"
                match_level = "HIGH"

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
                "match_level": match_level,
                "reason": f"Tech Match: {res['keyword_raw']*100:.0f}%, Vector: {res['vector_norm']:.2f}",
                
                # [Legacy Support] api/routes.py 호환
                "raw_score": final_score, 
                "is_exact_match": (note == "Best Match") or (final_score >= 85),
                
                # 내부 로직용 필드 유지 (feedback 용)
                "tech_stack": res['tech_stack'],
                "note": note,
                "keyword_raw": res['keyword_raw'],
                "vector_norm": res['vector_norm']
            })

        # 7. 피드백 생성
        report = self.generate_xai_feedback(resume_input, formatted_results)

        return formatted_results, report

# Singleton Instance
resume_engine = MatchingEngine()
