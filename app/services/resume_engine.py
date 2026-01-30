import json
import numpy as np
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv

# [핵심] .env 파일 로드를 위한 라이브러리
try:
    from dotenv import load_dotenv
except ImportError:
    print("⚠️ 'python-dotenv' 라이브러리가 없습니다. 'pip install python-dotenv'를 실행해주세요.")
    # 더미 함수 정의 (에러 방지)
    def load_dotenv(dotenv_path=None): pass

# 필수 라이브러리 로드 체크
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
except ImportError:
    print("⚠️ 필수 라이브러리가 설치되지 않았습니다. requirements.txt를 확인하세요.")
    
# ==========================================
# 0. 환경 변수(.env) 로드 및 진단
# ==========================================
print("\n[System] Loading environment variables...")
env_path = Path.cwd() / ".env" # 현재 위치에서 .env 찾기

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Found .env file at: {env_path}")
else:
    load_dotenv() # 기본 경로 탐색
    print("ℹ️ No explicit .env file found in root. Searching default locations...")



# ==========================================
# 1. 유틸리티 및 경로 설정 (Infrastructure)
# ==========================================
def get_data_path() -> Path:
    cwd = Path.cwd()
    # 다양한 경로 시도
    candidates = [
        cwd / "app" / "data",
        Path(__file__).parent / "app" / "data",
        Path(__file__).parent.parent / "app" / "data", # 상위 폴더 고려
        cwd / "services" / "data", # services 폴더 구조 대응
        cwd / "data"
    ]
    
    for path in candidates:
        if path.exists():
            return path
            
    # 못 찾으면 생성 시도 (에러 방지)
    try:
        (cwd / "data").mkdir(exist_ok=True)
        return cwd / "data"
    except (FileNotFoundError, PermissionError) as e:
        print(f"⚠️ Failed to create data directory: {e}")
        return cwd / "app" / "data"

class DataLoader:
    def __init__(self):
        self.base_path = get_data_path()
        self.file_names = {
            "resumes": "final_resume_600.json",
            "companies": "company_50_pool.json",
            "metadata": "final_metadata_600.json",
            "seeds": "final_jd_seeds_fixed.json" 
        }

# ==========================================
# 2. 매칭 엔진 클래스 (Core Logic)
# ==========================================
class MatchingEngine:
    # --------------------------------------
    # 상수 및 설정 (Configuration)
    # --------------------------------------
    WEIGHT_VECTOR = 0.55       # S-BERT 유사도 가중치
    WEIGHT_KEYWORD = 0.35      # 스킬 매칭 가중치
    BONUS_ROLE_MATCH = 0.10    # 직무 일치 보너스 (Max)
    GAP_THRESHOLD = 50.0       # 기술 갭(Skill Gap) 판단 기준 점수

    # 등급별 점수 구간 (엄격 적용)
    SCORE_RANGES = [
        (88.0, 97.0), # S (Rank 1)
        (77.0, 86.0), # A (Rank 2)
        (66.0, 75.0)  # B (Rank 3)
    ]

    # 티어별 추천 쿼터 (참고용)
    TIER_RULES = {
        "S": ["Top", "Top", "Mid"],
        "A": ["Top", "Mid", "Mid"],
        "B": ["Mid", "Mid", "Low"],
        "C": ["Mid", "Low", "Low"],
        "F": ["Low", "Low", "Low"]
    }
    # [NEW] 현실적인 Raw Score 기준점 (정규화 후 기준)
    RAW_SCORE_MIN = 0.30
    RAW_SCORE_MAX = 0.95
    GAP_THRESHOLD_RATIO = 0.50  # 0.5점(50점) 미만이면 경고


    def __init__(self):
        print("⚙️ Initializing Matching Engine (Dual List Applied)...")
        self.loader = DataLoader()
        self.base_path = self.loader.base_path
        
        # 1. OpenAI 초기화
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print("✅ OpenAI Client Connected")
            except Exception as e:
                print(f"⚠️ OpenAI Connection Failed: {e}")
        else:
            print("⚠️ No OPENAI_API_KEY found")

        # 2.0 Gemini 초기화 (Backup)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_client = None
        if self.google_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.google_api_key)
                print("✅ Google Gemini Client Connected (Backup - gemini-2.0-flash)")
            except Exception as e:
                print(f"⚠️ Google Gemini Connection Failed: {e}")
        
        # 2. 모델 로드
        try:
            # User Modified Model
            self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
            print("[Start] Matching Engine initializing...")
        except Exception as e:
            print(f"⚠️ Model Load Failed: {e}")
            self.model = None

        # 3. 데이터 로드 (캐싱 적용)
        loaded_data = self._load_company_vectors()
        if loaded_data:
            self.companies = loaded_data['companies']
            self.company_vectors = loaded_data['vectors']
            print(f"✅ Engine Ready: {len(self.companies)} companies loaded.")
        else:
            self.companies = []
            self.company_vectors = None

    # --------------------------------------
    # 내부 함수: 데이터 로딩 및 벡터화
    # --------------------------------------
    def _load_company_vectors(self):
        """PKL 캐시 우선 로드, 없으면 JSON 빌드"""
        pkl_path = self.base_path / "company_jd_vectors.pkl"
        json_path = self.base_path / "company_50_pool.json"
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and 'vectors' in data:
                    return data
            except (FileNotFoundError, PermissionError, Exception) as e:
                print(f"⚠️ Failed to load pickle cache: {e}")

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    companies = json.load(f)
                
                if not self.model: return {'companies': companies, 'vectors': None}

                texts = []
                for comp in companies:
                    # 기업 정보 텍스트화 (중요: 모든 필드 포함)
                    parts = [
                        comp.get('name', ''),
                        comp.get('industry', ''),
                        " ".join(comp.get('tech_stack', [])),
                        " ".join(comp.get('target_roles', [])),
                        comp.get('location', '')
                    ]
                    texts.append(" ".join(parts))
                
                vectors = self.model.encode(texts, show_progress_bar=False, batch_size=32)
                
                # 캐시 저장
                try:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump({'companies': companies, 'vectors': vectors}, f)
                except (FileNotFoundError, PermissionError) as e:
                    print(f"⚠️ Failed to save pickle cache: {e}")
                
                return {'companies': companies, 'vectors': vectors}
            except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
                print(f"⚠️ Failed to load company data: {e}")
                return None
        return None

    # --------------------------------------
    # 내부 함수: 텍스트 및 점수 처리 (Utils)
    # --------------------------------------
    def _normalize_text(self, text: str) -> str:
        """대소문자 통일 및 특수문자 제거"""
        if not text: return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9가-힣\s]', '', text)
        return text.strip()

    def _normalize_vector_score(self, val: float) -> float:
        """S-BERT 점수 정규화 (0.15~0.75 -> 0.0~1.0)"""
        min_bound = 0.15
        max_bound = 0.75
        normalized = (val - min_bound) / (max_bound - min_bound)
        return max(0.0, min(1.0, normalized))

    def get_grade(self, score: float) -> str:
        """점수 기반 등급 판정"""
        if score >= 88: return "S"
        if score >= 78: return "A"
        if score >= 68: return "B"
        if score >= 58: return "C"
        return "F"
    
    def _map_score_to_range(self, raw_score: float, target_min: float, target_max: float) -> float:
        """
        [Dynamic Scaling] 현실적인 입력 범위(Raw Score)를 목표 범위로 매핑
        """
        input_min, input_max = self.RAW_SCORE_MIN, self.RAW_SCORE_MAX

        normalized = (raw_score - input_min) / (input_max - input_min)
        normalized = max(0.0, min(1.0, normalized))

        scaled_score = target_min + (normalized * (target_max - target_min))
        return round(scaled_score, 1)

    def _convert_resume_to_text(self, resume_input: Dict) -> str:
        """이력서 객체를 텍스트로 변환"""
        content = resume_input.get('resume_content', {})
        text_parts = []

        # 1. raw_text (원본 텍스트)가 있으면 최우선적으로 포함
        if isinstance(content, dict) and content.get('raw_text'):
            text_parts.append(content['raw_text'])

        # 2. target_role (희망 직무) 추가
        target_role = resume_input.get('target_role', '')
        if target_role:
            text_parts.append(target_role)
        
        if isinstance(content, dict):
            # 3. skills (기술 스택) 추출
            skills = content.get('skills', {})
            user_skills = []
            if isinstance(skills, dict):
                user_skills = skills.get('essential', []) + skills.get('additional', [])
            elif isinstance(skills, list):
                user_skills = skills
            if user_skills:
                text_parts.append(" ".join(user_skills))
            
            # 4. professional_experience (경력 사항) 추출 - 역할 및 주요 업무 포함
            exp_texts = []
            for exp in content.get('professional_experience', []):
                if exp.get('role'): exp_texts.append(exp['role'])
                tasks = exp.get('key_tasks', [])
                if isinstance(tasks, list):
                    exp_texts.extend(tasks)
            if exp_texts:
                text_parts.append(" ".join(exp_texts))

            # 5. education (학력 사항) 추출 - 전공 및 학위 반영
            edu_texts = []
            for edu in content.get('education', []):
                if edu.get('major'): edu_texts.append(edu['major'])
                if edu.get('degree'): edu_texts.append(edu['degree'])
            if edu_texts:
                text_parts.append(" ".join(edu_texts))
        else:
            text_parts.append(str(content))
            
        # 모든 파트를 공백으로 구분하여 하나의 텍스트로 병합
        return " ".join([p for p in text_parts if p]).strip()

    # --------------------------------------
    # 내부 함수: 핵심 로직 (Logic)
    # --------------------------------------
    def _calculate_keyword_score(self, resume_text: str, tech_stack: List[str]) -> float:
        """기술 스택 커버리지 계산"""
        if not tech_stack: return 0.5
        resume_lower = resume_text.lower()
        match_count = sum(1 for skill in tech_stack if skill.lower() in resume_lower)
        return match_count / len(tech_stack)

    def _calculate_metadata_bonus(self, candidate_role: str, company_target_roles: List[str]) -> float:
        """직무 연관성 보너스 (Fullstack, AI, UI/UX 특화)"""
        if not candidate_role or not company_target_roles: return 0.0
        cand_role_lower = candidate_role.lower()
        
        for role in company_target_roles:
            role_lower = role.lower()
            if role_lower in cand_role_lower or cand_role_lower in role_lower:
                return self.BONUS_ROLE_MATCH
            if "fullstack" in cand_role_lower and (role_lower in ["backend", "frontend"]):
                return self.BONUS_ROLE_MATCH * 0.8
            if "ai" in cand_role_lower or "llm" in cand_role_lower:
                if any(kw in role_lower for kw in ["nlp", "llm", "vision", "ai", "ml", "data"]):
                    return self.BONUS_ROLE_MATCH
            if "ui/ux" in cand_role_lower or "designer" in cand_role_lower:
                if any(kw in role_lower for kw in ["design", "ui", "ux", "product", "creative"]):
                    return self.BONUS_ROLE_MATCH
        return 0.0

    def _check_role_relevance(self, target_category: str, current_role: str) -> bool:
        """직무 연관성 체크 (키워드 매칭)"""
        if not target_category or not current_role: return False
        target_category = target_category.lower()
        current_role = current_role.lower()
        
        relevance_keywords = [target_category, '기획', '개발', 'developer', 'manager', 'engineer', 'design']
        
        if "designer" in target_category or "ui/ux" in target_category:
            relevance_keywords.extend(['art', 'creative', 'ux', 'ui', '디자인', '디자이너', '퍼블리셔'])
        if "ai" in target_category or "llm" in target_category:
            relevance_keywords.extend(['researcher', 'scientist', 'nlp', 'ml', 'data', 'lab', '연구원'])
        
        return any(kw in current_role for kw in relevance_keywords)

    def _calculate_missing_skills(self, resume_input: Dict, company_stack: List[str]) -> List[str]:
        """부족한 스킬 도출"""
        if not company_stack: return []
        content = resume_input.get('resume_content', {})
        user_skills = []
        if isinstance(content, dict):
            skills = content.get('skills', {})
            if isinstance(skills, dict):
                user_skills = skills.get('essential', []) + skills.get('additional', [])
        
        user_norm = [s.lower().strip() for s in user_skills]
        missing = []
        for req in company_stack:
            req_l = req.lower().strip()
            if req_l in user_norm: continue
            if not any(req_l in u or u in req_l for u in user_norm):
                missing.append(req)
        return missing

    def _calculate_ats_detail(self, missing_skills: List[str], company_stack: List[str]) -> Dict[str, Any]:
        """ATS 점수 상세 계산"""
        total = len(company_stack)
        if total == 0: return {"score": 100, "matched": 0, "total": 0}
        matched = total - len(missing_skills)
        return {"score": int((matched/total)*100), "matched": matched, "total": total}

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

    # --------------------------------------
    # 기업 분류 및 점수 산정 로직 (Main Logic)
    # --------------------------------------
    def _categorize_companies(self, all_companies, vector_scores, resume_input, candidate_role):
        buckets = {"Top": [], "Mid": [], "Low": []}
        resume_text = self._convert_resume_to_text(resume_input)
        
        # 0. 이력서 기본 정보 추출
        content = resume_input.get('resume_content', {})
        experiences = content.get('professional_experience', [])
        
        actual_months = 0
        current_exp_role = ""
        if experiences:
            period_str = str(experiences[0].get('period', '0'))
            current_exp_role = experiences[0].get('role', '')
            nums = re.findall(r'\d+', period_str)
            actual_months = int(nums[0]) if nums else 0
            
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
            if meta_bonus == 0 and v_norm > 0.4:
                base_hybrid_score -= 0.15 
            
            # 5. [Ghost F 해결] 정밀 재채점
            final_raw_score = self.verify_and_regrade(resume_input, base_hybrid_score)

            # 6. [Team Rule] F등급 강제 판정 조건 강화
            is_forced_f = False
            f_reason = ""
            
            # (1) 무경력
            if actual_months == 0:
                is_forced_f = True
                f_reason = "실무 경력 없음(신입)"
            # (2) 직무 불일치
            elif not is_relevant_role:
                is_forced_f = True
                f_reason = "직무 연관성 부족"
            # (3) 기술 역량 미달
            elif k_score == 0 and v_norm < 0.3:
                is_forced_f = True
                f_reason = "핵심 기술 역량 부족"
            # (4) 이력서 내용 부족 (New)
            elif len(resume_text) < 50:
                is_forced_f = True
                f_reason = "이력서 내용 부족"

            if is_forced_f:
                # F등급 점수 제한 (최대 59점)
                final_raw_score = min(final_raw_score, 0.59)
            else:
                # C등급 이상 점수 보정 (최소 60점)
                if final_raw_score < 0.60:
                    final_raw_score = 0.60
            
            # 100점 만점 환산
            final_score_100 = final_raw_score * 100

            # 7. 부족한 스킬 계산
            missing_skills = self._calculate_missing_skills(resume_input, comp.get('tech_stack', []))
            
            comp_data = {
                "metadata": {
                    "company_name": comp["name"],
                    "job_title": ", ".join(comp.get("target_roles", [])),
                    "industry": comp["industry"],
                    "tier": comp.get("tier", "Low")
                },
                "tech_stack": comp.get("tech_stack", []),
                "raw_score": final_score_100,
                "vector_raw": round(v_raw, 2),
                "vector_norm": round(v_norm, 2),
                "keyword_raw": round(k_score, 2),
                "meta_bonus": round(meta_bonus, 2),
                "is_forced_f": is_forced_f,
                "f_reason": f_reason,
                "missing_skills": missing_skills
            }

            tier = comp.get("tier", "Low")
            if tier not in buckets: tier = "Low"
            buckets[tier].append(comp_data)

        for t in buckets:
            buckets[t].sort(key=lambda x: x['raw_score'], reverse=True)

        return buckets

    # --------------------------------------
    # [New] AI 피드백 생성 (Advanced Feedback)
    # --------------------------------------
    def _generate_rule_based_feedback(self, resume_input: dict, recommendations: List[Dict]) -> str:
        """
        전문적이고 객관적인 AI 피드백 생성
        - 등급별 톤 조절 및 상세 분석
        - [수정] 기업 매칭 결과를 맨 뒤로 이동
        - [수정] 직무 연관성을 수치 대신 정성적 표현(최적합/적당하다/부족하다)으로 변경
        """
        feedback_lines = ["\n종합 AI 코치 의견:"]

        if not recommendations:
            feedback_lines.append("분석 결과, 현재 이력서에 부합하는 추천 기업을 식별할 수 없습니다. 이력서의 기술 스택 및 경력 기술을 재점검하여 주시기 바랍니다.")
            return "\n".join(feedback_lines)

        # 1. 기본 정보 추출
        top_rec = recommendations[0]
        # 추천 리스트 구조 호환성 (metadata 안에 있거나 top level에 있거나)
        top_company_name = top_rec.get('metadata', {}).get('company_name') or top_rec.get('company_name', '추천 기업')
        top_score = top_rec['match_score']
        # note는 match_type으로 매핑
        top_note = top_rec.get('match_type', 'Good Fit')
        
        classification = resume_input.get('classification', {})
        predicted_role = classification.get('predicted_role') or resume_input.get('target_role', '미지정 직무')
        
        content = resume_input.get('resume_content', {})
        skills_data = content.get('skills', {})
        resume_all_skills = set()
        if isinstance(skills_data, dict):
             resume_all_skills = set(skills_data.get('essential', [])).union(set(skills_data.get('additional', [])))

        experiences = content.get('professional_experience', [])
        projects = content.get('project_experience', [])

        # ========================================================
        # [순서 변경] 상세 분석 내용을 먼저 구성하고, 매칭 결과는 맨 뒤로 보냅니다.
        # ========================================================

        # 2. [직무 적합도 상세 분석] 섹션
        job_fit_lines = ["\n[직무 적합도 상세 분석]"]
        
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
        job_fit_lines.append(f"- 기술 역량: {tech_fit_msg}")

        # 실무 경험 분석
        exp_count = len(experiences)
        if exp_count >= 1:
            exp_role = experiences[0].get('role', '관련 직무')
            exp_fit_msg = f"{exp_role} 경력을 통한 실무 기여 가능성이 높음으로 분석됩니다."
        else:
            exp_fit_msg = "실무 경력 증빙이 부족하여, 프로젝트 경험을 통한 역량 증명이 요구됩니다."
        job_fit_lines.append(f"- 실무 경험: {exp_fit_msg}")

        # [수정] 직무 연관성 분석 - 퍼센트 대신 정성적 표현 사용
        relevance_pct = int(top_rec.get('vector_norm', 0) * 100)
        if relevance_pct >= 80:
            relevance_term = "최적합 수준"
        elif relevance_pct >= 60:
            relevance_term = "적당한 수준"
        else:
            relevance_term = "다소 부족한 수준"
        
        job_fit_lines.append(f"- 직무 연관성: 지원하신 직무와 보유하신 경력 간의 연관성은 {relevance_term}으로 분석됩니다.")

        # 3. [AI 보강 제안] 섹션
        proposal_lines = ["\n[AI 보강 제안]"]
        
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

        proposal_lines.extend(proposals)

        # 4. [기업 매칭 결과] 섹션 (맨 뒤로 이동)
        match_result_lines = ["\n[기업 매칭 결과]"]
        match_result_lines.append(f"대상 기업: {top_company_name}")
        match_result_lines.append(f"평가 등급: {top_note} ({top_score}점)")
        
        tone_summary = "긍정적" if top_score >= 76 else "분석적"
        match_result_lines.append(f"분석 요약: 해당 이력서는 {predicted_role} 포지션에 대해 {tone_summary}인 정합성을 보이고 있습니다.")

        # 최종 조합 (순서: 직무 적합도 -> 보강 제안 -> 기업 매칭 결과)
        feedback_lines.extend(job_fit_lines)
        feedback_lines.extend(proposal_lines)
        feedback_lines.extend(match_result_lines)

        return "\n".join(feedback_lines)

    # ==========================================
    # 메인 메소드 (FastAPI 호환)
    # ==========================================
    def _ask_llm(self, prompt: str) -> str:
        """OpenAI와 Gemini를 모두 시도하는 Fallback LLM 호출기"""
        
        # 1. Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "당신은 전문 커리어 코치입니다. 한국어로 답변하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ OpenAI request failed: {e}")
        
        # 2. Try Gemini (Fallback)
        # 2. Try Gemini (Fallback)
        if self.gemini_client:
            print("🔄 Attempting fallback to Google Gemini...")
            try:
                response = self.gemini_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini request failed: {e}")
        else:
            print("⚠️ Fallback skipped: Google Gemini client is not initialized.")
        
        return "" # Both failed

    def generate_xai_feedback(self, resume_input: dict, recommendations: List[Dict]) -> str:
        """
        전문적이고 객관적인 AI 피드백 생성 (LLM Enhancement)
        - Rule-based 초안 생성 후 LLM으로 세련되게 다듬기
        """
        feedback_lines = ["\n종합 AI 코치 의견:"]

        if not recommendations:
            feedback_lines.append("분석 결과, 현재 이력서에 부합하는 추천 기업을 식별할 수 없습니다. 이력서의 기술 스택 및 경력 기술을 재점검하여 주시기 바랍니다.")
            return "\n".join(feedback_lines)

        # 1. 기본 정보 추출 (생략)
        top_rec = recommendations[0]
        top_company_name = top_rec.get('metadata', {}).get('company_name') or top_rec.get('company_name', '추천 기업')
        top_score = top_rec['match_score']
        top_note = top_rec.get('match_type', 'Good Fit')
        
        classification = resume_input.get('classification', {})
        predicted_role = classification.get('predicted_role') or resume_input.get('target_role', '미지정 직무')
        
        content = resume_input.get('resume_content', {})
        skills_data = content.get('skills', {})
        resume_all_skills = set()
        if isinstance(skills_data, dict):
             resume_all_skills = set(skills_data.get('essential', [])).union(set(skills_data.get('additional', [])))

        experiences = content.get('professional_experience', [])
        projects = content.get('project_experience', [])

        # 2. Rule-based 초안 생성
        draft_lines = []
        
        # 기술 역량 분석 (Raw Logic)
        tech_match_pct = int(top_rec.get('keyword_raw', 0) * 100)
        skills_list = list(resume_all_skills)
        draft_lines.append(f"기술 일치도: {tech_match_pct}%")
        
        # 실무 경험 분석
        draft_lines.append(f"경력: {len(experiences)}건")
        
        # 직무 연관성
        relevance_pct = int(top_rec.get('vector_norm', 0) * 100)
        draft_lines.append(f"직무 연관성: {relevance_pct}%")
        
        # 3. LLM에게 다듬기 요청
        context = "\n".join(draft_lines)
        prompt = f"""
        지원자 정보: {context}
        추천 기업: {top_company_name} ({top_score}점)
        직무: {predicted_role}

        위 정보를 바탕으로, 지원자에게 해줄 수 있는 '종합 피드백'을 3~4문장으로 작성해줘.
        1. 기술 역량에 대한 객관적 평가
        2. 강점과 보완점
        3. 격려 및 조언
        말투는 "전문 컨설턴트"처럼 정중하게 해줘.
        """
        
        llm_feedback = self._ask_llm(prompt)
        
        if llm_feedback:
             return f"\n종합 AI 코치 의견:\n{llm_feedback}"
        else:
             # Fallback to rule-based logic (if LLM fails)
             return self._generate_rule_based_feedback(resume_input, recommendations)



    # ==========================================
    # 메인 메소드 (FastAPI 호환)
    # ==========================================
    def recommend(self, resume_input: dict) -> Tuple[List[Dict], str]:
        """FastAPI 라우터 호환용 메인 메소드"""
        # [방어 코드] 기업 데이터 확인
        if not self.companies or self.company_vectors is None:
            return [], "시스템 에러: 기업 데이터(Vector DB)가 로드되지 않았습니다."

        # 1. 이력서 텍스트 변환
        resume_text = self._convert_resume_to_text(resume_input)
        
        # 2. 직무 파악
        classification = resume_input.get('classification', {})
        role = classification.get('predicted_role') or resume_input.get('target_role', 'backend')

        # 3. 벡터 임베딩 및 유사도 계산
        if self.model:
            query_vector = self.model.encode([resume_text])[0]
            vector_scores = cosine_similarity([query_vector], self.company_vectors)[0]
        else:
            vector_scores = [0.0] * len(self.companies)

        # 4. 버킷팅 및 점수 계산 (Main Scoring)
        buckets = self._categorize_companies(self.companies, vector_scores, resume_input, role)

        # 5. 등급 기반 기업 선정 (Smart Selection)
        all_comp_data = []
        for t in buckets: all_comp_data.extend(buckets[t])
        
        # 강제 F등급 여부 확인
        is_candidate_forced_f = any(c.get('is_forced_f', False) for c in all_comp_data)
        
        resume_evaluation = resume_input.get('evaluation') or {}
        candidate_grade = resume_evaluation.get('grade')
        
        if is_candidate_forced_f:
            candidate_grade = "F"
        elif candidate_grade is None:
            if all_comp_data:
                # 상위 3개 평균으로 자동 등급 산정
                sorted_companies = sorted(all_comp_data, key=lambda x: x['raw_score'], reverse=True)
                top_scores = [c['raw_score'] for c in sorted_companies[:3]]
                avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
                candidate_grade = self.get_grade(avg_score)
            else:
                candidate_grade = "B"
        elif candidate_grade == "F":
            # Ghost F Recovery (구제 로직)
            all_raw = [c['raw_score'] for c in all_comp_data]
            if all_raw and (sum(all_raw)/len(all_raw)) > 60:
                candidate_grade = "C"

        # 타겟 티어 선정
        target_slots = self.TIER_RULES.get(candidate_grade, self.TIER_RULES["B"])
        final_selection = []
        used_companies = set()

        for required_tier in target_slots:
            selected = None
            for comp in buckets.get(required_tier, []):
                if comp['metadata']['company_name'] not in used_companies:
                    selected = comp
                    break
            
            # Fallback (해당 티어에 없으면 다른 티어 검색)
            if not selected:
                order = ["Top", "Mid", "Low"] if required_tier == "Top" else ["Mid", "Low", "Top"]
                for tier in order:
                    for comp in buckets.get(tier, []):
                        if comp['metadata']['company_name'] not in used_companies:
                            selected = comp
                            break
                    if selected: break
            
            if selected:
                used_companies.add(selected['metadata']['company_name'])
                final_selection.append(selected)

        # 6. 점수 매핑 및 포맷팅
        formatted_results = [] # 응답용 (Slim)
        full_data_results = [] # 내부용 (Full)

        for i, res in enumerate(final_selection):
            # Dynamic Scaling
            if i < len(self.SCORE_RANGES):
                min_s, max_s = self.SCORE_RANGES[i]
                final_score = self._map_score_to_range(res['raw_score'], min_s, max_s)
            else:
                final_score = round(res['raw_score'], 1)

            if i == 0:
                note = "Best Match"
                match_level = "BEST"
            elif res['raw_score'] < self.GAP_THRESHOLD_RATIO:
                note = "Skill Gap"
                match_level = "GAP"
            else:
                note = "High Fit"
                match_level = "HIGH"

            # ATS 상세 정보 역산 (Full 데이터용)
            ats_data = self._calculate_ats_detail(res.get('missing_skills', []), res.get('tech_stack', []))

            # 1. 내부용 Full Data (피드백 생성에 필요)
            full_entry = {
                "metadata": res['metadata'],
                "company_name": res['metadata']['company_name'],
                "match_score": final_score,
                "tier": res['metadata']['tier'],
                "match_type": note,
                "match_level": match_level,
                "reason": f"Tech Match: {res['keyword_raw']*100:.0f}%, Vector: {res['vector_norm']:.2f}",
                "raw_score": final_score,
                "is_exact_match": (note == "Best Match") or (final_score >= 85),
                "tech_stack": res['tech_stack'],
                "missing_skills": res.get('missing_skills', []),
                "note": note,
                "keyword_raw": res['keyword_raw'],
                "vector_norm": res['vector_norm'],
                "ats_score": ats_data
            }
            full_data_results.append(full_entry)

            # 2. 응답용 Slim Data (7개 필드 제거됨)
            slim_entry = {
                "company_name": res['metadata']['company_name'],
                "match_score": final_score,
                "tier": res['metadata']['tier'],
                "match_type": note,
                "match_level": match_level,
                "missing_skills": res.get('missing_skills', []),
                "note": note,
                "is_exact_match": (note == "Best Match") or (final_score >= 85)
            }
            formatted_results.append(slim_entry)

        # 7. AI 피드백 생성 (Full Data 사용!)
        report = self.generate_xai_feedback(resume_input, full_data_results)

        # [수정] formatted_results 대신 full_data_results를 반환합니다.
        # 이유: main.py가 tech_stack, reason 등을 필수 필드로 요구하기 때문에
        # formatted_results를 보내면 500 에러가 발생합니다.
        # 화면에도 Tech Stack을 보여주려면 Full Data가 필요합니다.
        return full_data_results, report

if __name__ == "__main__":
    engine = MatchingEngine()
    print("Engine Fully Initialized with All Features.")