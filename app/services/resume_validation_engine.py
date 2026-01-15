import json
import re
import os
from pathlib import Path

# ==========================================
# 1. 데이터 로드 (경로 및 로딩 확인 강화)
# ==========================================
class DataLoader:
    def __init__(self):
        # 1. 데이터 파일이 위치한 실제 절대 경로 설정
        # 선생님의 환경에 맞춰 백슬래시 에러 방지를 위해 r"" (raw string) 사용
        self.base_path = Path(r"C:\TheCareer\NextEnterAI\app\data")
        
        # 2. 파일명 정의 (실제 폴더의 파일명과 대소문자까지 일치해야 합니다)
        self.file_names = {
            "resumes": "final_resume_600.json",
            "companies": "company_50_pool.json",
            "metadata": "final_metadata_600.json"
        }
        
        self.data = {}
        self._load_all_files()

        # 빠른 조회를 위한 맵 생성
        metadata_list = self.data.get("metadata", [])
        self.metadata_map = {item['resume_id']: item for item in metadata_list if 'resume_id' in item}

    def _load_all_files(self):
        """
        지정된 경로에서 JSON 파일들을 읽어오고 로드 결과를 출력합니다.
        """
        # 경로 존재 여부 확인
        if not self.base_path.exists():
            print(f"❌ [경로 에러] 폴더를 찾을 수 없습니다: {self.base_path}")
            # 폴더가 없으면 현재 실행 위치(CWD)에서 찾는 것으로 우회
            self.base_path = Path.cwd()
            print(f"ℹ️ [우회] 현재 작업 디렉토리에서 파일을 찾습니다: {self.base_path}")

        for key, filename in self.file_names.items():
            path = self.base_path / filename
            
            try:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                        self.data[key] = loaded_data
                    # 로드 성공 시 개수를 출력하여 "못 읽었는지" 바로 확인 가능하게 함
                    print(f"✅ 로드 완료: {filename} ({len(loaded_data)}개 레코드) -> 경로: {path}")
                else:
                    print(f"⚠️ 파일을 찾을 수 없습니다: {filename} (확인 필요 경로: {path})")
                    self.data[key] = []
            except Exception as e:
                print(f"❌ {filename} 로딩 중 에러 발생: {e}")
                self.data[key] = []

    def normalize(self, text):
        """
        기술 스택 비교를 위해 소문자 변환 및 특수문자 제거
        """
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9]', '', str(text).lower())

# ==========================================
# 2. 매칭 엔진 (Python 논리 기반 추천)
# ==========================================
class MatchingEngine:
    def __init__(self):
        self.dl = DataLoader()
        # 데이터가 비어있을 경우를 대비해 초기값 설정
        self.companies = self.dl.data.get("companies", [])
        self.base_path = self.dl.base_path

    def calculate_score(self, resume_skills, company_stack):
        """
        논문 기반 ATS 공식 ($S_{matched} / S_{required}$) 구현
        """
        if not company_stack: return 0.0
        
        r_set = {self.dl.normalize(s) for s in resume_skills if s}
        c_set = {self.dl.normalize(s) for s in company_stack if s}
        
        matched = r_set.intersection(c_set)
        
        if not c_set: return 0.0
        return round((len(matched) / len(c_set)) * 100, 1)

    def recommend(self, resume_input: dict):
        """
        FastAPI 라우터나 테스트 스크립트에서 호출하는 메인 추천 함수
        """
        # [방어 코드] 기업 데이터가 아예 안 읽혔을 경우 처리
        if not self.companies:
            print("❌ [추천 중단] 기업 데이터(companies)가 로드되지 않았습니다.")
            return [], "시스템 에러: 기업 데이터 풀이 비어있습니다. JSON 파일 위치를 확인하세요."

        # 1. 데이터 추출 및 구조 파악
        # (1) AI 분석 결과(classification)가 있는 경우
        classification = resume_input.get('classification', {})
        role = classification.get('predicted_role', '').lower()
        skills = classification.get('keywords', [])
        
        # (2) AI 분석 결과가 아닌 일반 DTO(target_role 등) 구조인 경우 처리
        if not skills:
            # resume_content 내부의 skills 탐색
            content = resume_input.get('resume_content', {})
            skills_info = content.get('skills', {})
            skills = skills_info.get('essential', []) + skills_info.get('additional', [])
            
            # 직무명 결정 (우선순위: standardized_role > target_role)
            role = resume_input.get('standardized_role', {}).get('category', '')
            if not role:
                role = resume_input.get('target_role', 'backend')
        
        role = role.lower()
        scored_list = []
        
        # 추천 프로세스 로그
        print(f"🔍 [추천 엔진 가동] 분석 직무: '{role}', 추출 기술: {len(skills)}개")
        
        for comp in self.companies:
            # 2. 직무 필터링 (Role-Based Filtering)
            # 기업 공고의 target_roles와 지원자의 role이 부분 일치하는지 확인
            target_roles = [r.lower() for r in comp.get('target_roles', [])]
            
            is_role_match = False
            for target in target_roles:
                # 'backend'와 'Backend Developer'가 서로를 포함하면 매칭으로 인정
                if target in role or role in target:
                    is_role_match = True
                    break
            
            if not is_role_match:
                continue
            
            # 3. 기술 점수 계산 (Skill-Based Scoring)
            tech_score = self.calculate_score(skills, comp.get('tech_stack', []))
            
            scored_list.append({
                "metadata": {
                    "company_name": comp['name'],
                    "job_title": role.upper(),
                    "tier": comp.get('tier', 'Mid')
                },
                "raw_score": tech_score,
                "is_exact_match": tech_score >= 85
            })

        # 4. 결과 정렬 및 가공
        scored_list.sort(key=lambda x: x['raw_score'], reverse=True)
        top_3 = scored_list[:3]
        
        # 5. 리포트 메시지 생성
        if not top_3:
            report = f"분석된 직무('{role}')와 일치하는 기업을 찾지 못했습니다. 기술 스택이나 직무명을 확인해 보세요."
        else:
            report = f"분석된 {role} 역량을 바탕으로 총 {len(scored_list)}개 후보 중 상위 {len(top_3)}곳을 추천합니다."
            report += f" 가장 적합한 곳은 {top_3[0]['metadata']['company_name']}으로, 기술 일치율은 {top_3[0]['raw_score']}%입니다."

        return top_3, report

# FastAPI 서비스에서 임포트할 인스턴스
resume_validation_engine = MatchingEngine()