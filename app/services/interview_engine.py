import re
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from app.services.file_parser import FileParser

class InterviewEngine:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ GOOGLE_API_KEY not found. Gemini features will be disabled.")
            self.client = None
        else:
            try:
                # Initialize Client with explicit API key
                self.client = genai.Client(api_key=api_key)
                print("✅ Gemini Integration Initialized (Model: gemini-2.0-flash)")
            except Exception as e:
                print(f"⚠️ Gemini Connection Error: {e}")
                import traceback
                traceback.print_exc()
                self.client = None

        # State Management
        self.chat_history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        
        # Phase Management (INTRO -> MAIN -> CLOSING)
        self.current_phase: str = "INTRO"  # INTRO, MAIN, CLOSING
        self.current_topic_probe_count: int = 0
        self.max_probes_per_topic: int = 2  # 같은 주제에 대해 최대 2번까지만 추가 질문
        self.topics_covered: List[str] = []  # 다룬 주제들 (프로젝트명 등)

    def _call_llm(self, prompt: str) -> str:
        if not self.client:
            return ""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini Generation Error: {e}")
            return ""

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON from LLM response, handling markdown code blocks."""
        try:
            json_str = text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse JSON from LLM: {text[:100]}...")
            return {}

    def normalize_role(self, target_role: Optional[str]) -> str:
        return target_role or "backend"

    def _resume_summary_for_prompt(self, resume_content: Optional[Dict[str, Any]]) -> str:
        """구조화 필드가 비어 있으면 raw_text를 우선 사용해 프롬프트용 요약 문자열 반환."""
        if not resume_content:
            return "No resume content provided."
        raw_text = resume_content.get("raw_text") or ""
        use_raw_primary = resume_content.get("_raw_text_primary") or False
        sk = resume_content.get("skills")
        skills_nonempty = (
            (isinstance(sk, list) and len(sk or []) > 0)
            or (isinstance(sk, dict) and (len((sk or {}).get("essential") or []) > 0 or len((sk or {}).get("additional") or []) > 0))
        )
        has_structure = (
            skills_nonempty
            or (isinstance(resume_content.get("education"), list) and len(resume_content.get("education") or []) > 0)
            or (isinstance(resume_content.get("professional_experience"), list) and len(resume_content.get("professional_experience") or []) > 0)
            or (isinstance(resume_content.get("project_experience"), list) and len(resume_content.get("project_experience") or []) > 0)
        )
        if (use_raw_primary or not has_structure) and raw_text:
            return f"Resume (raw text):\n\"\"\"\n{raw_text.strip()}\n\"\"\""
        return json.dumps(resume_content, ensure_ascii=False, indent=2)

    def build_seed_question(self, role: str, resume_content: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], portfolio_text: Optional[str] = None, difficulty: str = "JUNIOR", previous_questions: List[str] = []) -> Tuple[str, str, List[str]]:
        # Use LLM to generate a contextual seed question (raw_text fallback when structure is empty)
        resume_summary = self._resume_summary_for_prompt(resume_content)
        
        # Difficulty Adjustment
        difficulty_instruction = ""
        if difficulty == "SENIOR":
            difficulty_instruction = "Assess ARCHITECTURE design, scalability, trade-offs, and leadership skills. Ask complex, high-level technical questions."
        else:
            difficulty_instruction = "Assess FUNDAMENTALS, potential, and problem-solving basics. Ask approachable but technically valid questions."
            
        previous_context = ""
        if previous_questions:
            previous_context = f"AVOID repeating these previously asked questions:\n" + "\n".join([f"- {q}" for q in previous_questions])

        prompt = f"""
        You are a technical interviewer for a {role} position.
        Difficulty Level: {difficulty}
        Instruction: {difficulty_instruction}
        
        This is a follow-up question during the interview. Build upon prior context if available.
        
        Resume Summary:
        {resume_summary}

        Portfolio Summary:
        {json.dumps(portfolio, ensure_ascii=False, indent=2)}
        
        Portfolio Parsed Content (PDF/Docx):
        \"\"\"{portfolio_text or "No attached portfolio files."}\"\"\"

        Constraints:
        {previous_context}

        Task:
        Generate an interview question in Korean that explores ONE of these areas (VARY your choice each time):
        1. **Project Experience**: A specific project from their resume - technical challenges, solutions, outcomes
        2. **Professional/Career Experience**: Their role at a company, team collaboration, leadership, or organizational contributions
        3. **Technical Skills**: Deep-dive into a specific technology or skill they claim expertise in
        
        Requirements:
        - Reference a SPECIFIC item from the candidate's resume (project name, company name, or skill)
        - Ask them to explain with specific situation, their role, actions taken, and outcomes
        - **CRITICAL: NEVER use the words "STAR", "STARR", "스타", or "STAR 방식" in your question. Just ask naturally in Korean without mentioning any methodology names.**
        - Matches the {difficulty} level complexity
        - For SENIOR: Focus on architecture decisions, leadership, and strategic impact
        - For JUNIOR: Focus on learning experience, problem-solving approach, and growth
        - DO NOT ask the same question twice.
        
        Example formats:
        - Project: "이력서에 [프로젝트명] 프로젝트가 있는데, 이 프로젝트에서 맡으신 역할과 어떤 기술적 도전이 있었는지, 그리고 결과는 어땠는지 구체적으로 말씀해 주세요."
        - Career: "[회사명]에서 [직책]으로 근무하시면서 가장 큰 성과를 낸 경험이 있으시다면, 당시 상황과 본인이 기여한 부분을 상세히 설명해 주시겠어요?"
        - Skill: "이력서에 [기술명]에 능숙하다고 하셨는데, 실제로 어떤 상황에서 이 기술을 활용해 문제를 해결하셨나요?"
        
        Output JSON:
        {{
            "question": "The interview question string in Korean",
            "probe_goal": "Short description of what you want to verify",
            "requested_evidence": ["impact metrics", "specific tech stack decision"]
        }}
        """
        
        response_text = self._call_llm(prompt)
        data = self._parse_json_response(response_text)
        
        return (
            data.get("question", "대표적인 프로젝트 경험에서 맡으신 역할과 어떻게 문제를 해결하셨는지 구체적으로 설명해 주세요."),
            data.get("probe_goal", "핵심 역량 확인"),
            data.get("requested_evidence", ["구체적 행동", "정량적 성과"])
        )

    def build_intro_question(self, role: str, resume_content: Optional[Dict[str, Any]], difficulty: str = "JUNIOR") -> Tuple[str, str, List[str]]:
        """Generate an introduction question (self-introduction, motivation)."""
        resume_summary = self._resume_summary_for_prompt(resume_content)
        
        tone_instruction = "Expected Tone: Encouraging and patient." if difficulty == "JUNIOR" else "Expected Tone: Professional and direct."
        
        prompt = f"""
        You are a friendly technical interviewer for a {role} position.
        Difficulty: {difficulty}. {tone_instruction}
        
        This is the VERY FIRST question - an ice-breaker to start the interview.
        
        Resume Summary:
        {resume_summary}

        Task:
        Generate an opening question in Korean that:
        1. Asks the candidate to briefly introduce themselves
        2. Asks about their motivation for applying to this {role} position
        3. Is warm and welcoming to reduce nervousness
        
        Example: "안녕하세요! 간단한 자기소개와 함께, {role} 포지션에 지원하게 된 동기를 말씀해 주시겠어요?"
        
        Output JSON:
        {{
            "question": "The introduction question in Korean",
            "probe_goal": "지원 동기 및 열정 확인",
            "requested_evidence": ["career motivation", "role fit"]
        }}
        """
        
        response_text = self._call_llm(prompt)
        data = self._parse_json_response(response_text)
        
        return (
            data.get("question", f"안녕하세요! 간단한 자기소개와 함께, {role} 포지션에 지원하게 된 동기를 말씀해 주시겠어요?"),
            data.get("probe_goal", "지원 동기 및 열정 확인"),
            data.get("requested_evidence", ["career motivation", "role fit"])
        )

    def build_closing_question(self, role: str, difficulty: str = "JUNIOR") -> Tuple[str, str, List[str]]:
        """Generate a closing question based on difficulty."""
        
        if difficulty == "SENIOR":
            closing_question = (
                "마지막 질문입니다. 만약 우리 회사의 기술 리더로서 합류하시게 된다면, "
                "가장 먼저 해결하고 싶은 기술적 과제나 도입하고 싶은 문화가 있으신가요? "
                
            )
            return (
                closing_question,
                "리더십 및 기술 비전 확인",
                ["technical vision", "leadership", "strategic thinking"]
            )
        else:
            # JUNIOR/Default
            closing_question = (
                "마지막 질문입니다. 우리 회사에 입사하시게 된다면, "
                "앞으로 어떤 방향으로 성장하고 기여하고 싶으신지 말씀해 주세요., "
                
            )
            return (
                closing_question,
                "성장 비전 및 회사 적합성 확인",
                ["growth mindset", "company fit", "curiosity"]
            )

    def analyze_answer(self, answer: str) -> Dict[str, Any]:
        """
        STARR 기반 답변 분석 (문맥 기반 - 확장 프롬프트)

        중요: 직접적인 키워드("상황은", "결과는")가 없어도
        문맥상 해당 요소가 설명되었다면 인정해야 함
        """
        prompt = f"""
        당신은 면접 답변 분석 전문가입니다. 아래 답변을 STARR 방법론에 따라 **관대하게** 분석하세요.

        ## 지원자 답변:
        "{answer}"

        ## ⚠️ 핵심 원칙: 문맥 기반 인식 (매우 중요!)

        각 STARR 요소는 **직접적인 키워드("상황은", "과제는", "결과는") 없이도**
        문맥상 해당 내용이 암시되거나 포함되면 **true로 인정**합니다.
        면접자는 학술적 용어를 쓰지 않고 자연스럽게 말합니다.

        ---

        ## 1. Situation (상황) - 인정 기준

        다음 중 **하나라도** 해당하면 situation = true:

        **시간/시기 표현:**
        - "당시", "그때", "때", "기간", "동안", "중에", "무렵", "시절"
        - "작년", "올해", "지난", "최근", "예전", "과거"
        - "학기", "학년", "개월", "주간"

        **장소/조직 표현:**
        - "프로젝트", "회사", "팀", "부서", "조직", "그룹"
        - "학교", "대학", "대학교", "부트캠프", "캠프"
        - "인턴", "인턴십", "현장실습", "실습"
        - "스타트업", "기업", "연구소", "연구실", "동아리"

        **맥락 표현:**
        - "에서", "에서는", "에서의", "중에", "때"
        - "처음", "시작", "초반", "입사", "합류"
        - "상황", "배경", "계기", "맥락", "환경"

        **예시:** "졸업 프로젝트에서", "인턴 때", "팀에서", "작년에" → situation = true

        ---

        ## 2. Task (과제) - 인정 기준

        다음 중 **하나라도** 해당하면 task = true:

        **역할/담당 표현:**
        - "담당", "맡", "역할", "책임", "임무", "업무"
        - "리드", "리더", "PM", "PL", "TL"
        - "프론트", "백엔드", "풀스택"

        **목표/과제 표현:**
        - "목표", "과제", "미션", "목적"
        - "요구사항", "요청", "니즈"
        - "해야", "필요", "하게 되", "맡게"

        **문제/이슈 표현:**
        - "문제", "이슈", "버그", "오류", "에러", "장애"
        - "개선", "수정", "최적화", "성능"

        **기능/대상 표현:**
        - "기능", "모듈", "컴포넌트", "페이지", "화면"
        - "API", "서비스", "시스템", "앱", "웹"

        **예시:** "로그인 기능을 만들어야", "성능 개선이 필요", "백엔드 담당" → task = true

        ---

        ## 3. Action (행동) - 인정 기준

        다음 중 **하나라도** 해당하면 action = true:

        **행동 동사 (과거형):**
        - "했습니다", "했어요", "했고", "하여", "해서"
        - "구현", "개발", "작성", "설계", "적용", "사용"
        - "배포", "테스트", "수정", "추가", "연동"

        **기술 스택 언급:**
        - React, Vue, Angular, Next.js, Spring, Django, FastAPI
        - Java, Python, JavaScript, TypeScript, Kotlin
        - MySQL, PostgreSQL, MongoDB, Redis
        - AWS, Docker, Kubernetes, Git
        - (기타 모든 기술명)

        **작업 설명:**
        - "코드", "API", "DB", "쿼리", "로직"
        - "협업", "소통", "회의", "리뷰"

        **예시:** "React로 구현", "Spring 사용", "API 개발", "테스트 작성" → action = true

        ---

        ## 4. Result (결과) - 인정 기준

        다음 중 **하나라도** 해당하면 result = true:

        **완료/성공 표현:**
        - "완료", "완성", "마무리", "성공"
        - "출시", "오픈", "배포", "릴리즈"
        - "해결", "처리", "달성"

        **성과/개선 표현:**
        - "결과", "성과", "효과", "영향"
        - "개선", "향상", "증가", "감소"
        - "빨라", "좋아", "나아"

        **수치/지표:**
        - 퍼센트(%), 배수(배), 인원(명), 건수(건)
        - "TPS", "응답시간", "트래픽"

        **예시:** "배포 완료", "30% 향상", "성능 개선", "출시 성공" → result = true

        ---

        ## 5. Reflection (성찰) - 인정 기준

        다음 중 **하나라도** 해당하면 reflection = true:

        **배움/깨달음:**
        - "배웠", "배운", "느꼈", "느낀", "깨달"
        - "알게", "이해하게", "경험"

        **성장/발전:**
        - "성장", "발전", "향상", "역량", "능력"

        **반성/부족:**
        - "부족", "아쉬", "어려", "힘들"

        **다짐/계획:**
        - "다음에는", "앞으로", "노력", "계획"

        **예시:** "이 경험을 통해 배웠다", "성장했다", "다음에는 더" → reflection = true

        ---

        ## 기여도 분석:
        - "clear": "저는", "제가", "I", "내가", "직접", "혼자" 사용
        - "mixed": "저희", "우리"와 "저" 혼용
        - "unclear": 주로 "우리", "팀" 위주 (개인 기여 불명확)

        ## 구체적 증거 추출:
        기술명, 수치, 기간, 성과 지표 등 구체적인 정보 추출

        ## Output JSON (반드시 이 형식으로):
        {{
            "starr": {{
                "situation": true 또는 false,
                "task": true 또는 false,
                "action": true 또는 false,
                "result": true 또는 false,
                "reflection": true 또는 false
            }},
            "contribution": "clear" 또는 "mixed" 또는 "unclear",
            "evidence_clips": ["증거1", "증거2"],
            "answer_quality": "excellent" 또는 "good" 또는 "fair" 또는 "poor"
        }}
        """

        response_text = self._call_llm(prompt)
        result = self._parse_json_response(response_text)

        # LLM 실패 시 규칙 기반 fallback 분석
        if not result or "starr" not in result:
            print("⚠️ LLM STARR analysis failed. Using rule-based fallback.")
            result = self._analyze_answer_fallback(answer)

        # 답변 품질 기본값 설정
        if "answer_quality" not in result:
            starr = result.get("starr", {})
            filled = sum(1 for v in starr.values() if v)
            if filled >= 4:
                result["answer_quality"] = "excellent"
            elif filled >= 3:
                result["answer_quality"] = "good"
            elif filled >= 2:
                result["answer_quality"] = "fair"
            else:
                result["answer_quality"] = "poor"

        return result

    def _analyze_answer_fallback(self, answer: str) -> Dict[str, Any]:
        """
        LLM 실패 시 규칙 기반 STARR 분석 (문맥 인식 - 확장판)

        직접 키워드가 아닌 의미적 패턴으로 인식:
        - Situation: 배경/상황/시기/장소 언급
        - Task: 목표/역할/해야 할 일 언급
        - Action: 행동/구현/사용 언급
        - Result: 결과/성과/완료 언급
        - Reflection: 배움/느낌/다짐 언급
        """
        if not answer:
            return {
                "starr": {"situation": False, "task": False, "action": False, "result": False, "reflection": False},
                "contribution": "unclear",
                "evidence_clips": [],
                "answer_quality": "poor"
            }

        # ========================================
        # Situation 패턴 (배경/상황/맥락/시기/장소)
        # ========================================
        situation_patterns = [
            # 시간/시기 표현
            r"당시", r"그때", r"때", r"기간", r"동안", r"중에", r"무렵", r"시절",
            r"년도", r"학기", r"학년", r"개월", r"주간", r"일간",
            r"작년", r"올해", r"지난", r"최근", r"예전", r"과거",
            r"초기", r"중반", r"후반", r"말", r"초",

            # 장소/조직 표현
            r"프로젝트", r"회사", r"팀", r"부서", r"조직", r"그룹",
            r"학교", r"대학", r"대학교", r"학원", r"교육", r"캠프", r"부트캠프",
            r"인턴", r"인턴십", r"현장실습", r"실습",
            r"스타트업", r"기업", r"법인", r"센터", r"연구소", r"연구실",
            r"동아리", r"학회", r"모임", r"커뮤니티",

            # 상황/맥락 표현
            r"상황", r"배경", r"계기", r"맥락", r"환경", r"여건", r"조건",
            r"에서", r"에선", r"에서는", r"에서의",
            r"처음", r"시작", r"초반", r"입사", r"입학", r"합류",
            r"당면", r"직면", r"마주", r"겪", r"발생", r"생기",

            # 프로젝트/업무 유형
            r"졸업", r"캡스톤", r"토이", r"사이드", r"개인", r"팀",
            r"외주", r"용역", r"SI", r"SM", r"운영", r"유지보수",
            r"신규", r"리뉴얼", r"마이그레이션", r"고도화",
        ]
        has_situation = any(re.search(p, answer) for p in situation_patterns)

        # ========================================
        # Task 패턴 (과제/역할/목표/책임)
        # ========================================
        task_patterns = [
            # 역할/담당 표현
            r"담당", r"맡", r"역할", r"책임", r"임무", r"업무", r"일",
            r"리드", r"리더", r"매니저", r"PM", r"PL", r"TL",
            r"메인", r"서브", r"보조", r"지원",
            r"프론트", r"백엔드", r"풀스택", r"데브옵스", r"QA", r"기획",

            # 목표/과제 표현
            r"목표", r"과제", r"미션", r"태스크", r"task", r"목적",
            r"요구사항", r"요청", r"니즈", r"needs",
            r"해야", r"필요", r"요구", r"원하", r"바라",
            r"하게 되", r"을 하게", r"를 하게", r"맡게",

            # 문제/이슈 표현
            r"문제", r"이슈", r"버그", r"오류", r"에러", r"장애",
            r"개선", r"보완", r"수정", r"변경", r"리팩토링", r"리팩터링",
            r"최적화", r"튜닝", r"성능", r"속도",

            # 기능/개발 대상
            r"기능", r"모듈", r"컴포넌트", r"페이지", r"화면", r"뷰",
            r"API", r"엔드포인트", r"서비스", r"시스템", r"플랫폼",
            r"앱", r"어플", r"웹", r"사이트", r"포털",
            r"만들", r"구현해야", r"개발해야", r"작성해야",
        ]
        has_task = any(re.search(p, answer) for p in task_patterns)

        # ========================================
        # Action 패턴 (행동/실행/구현/기술 사용)
        # ========================================
        action_patterns = [
            # 행동 동사 (과거형 포함)
            r"했습니다", r"했어요", r"했고", r"했는데", r"했으며",
            r"하여", r"해서", r"하고", r"함으로써",
            r"했던", r"한 적", r"해본", r"해봤",
            r"진행", r"수행", r"실행", r"실시", r"시행",

            # 개발/구현 동사
            r"구현", r"개발", r"작성", r"코딩", r"프로그래밍",
            r"설계", r"디자인", r"아키텍처", r"구조화",
            r"적용", r"도입", r"채택", r"사용", r"활용", r"이용",
            r"연동", r"통합", r"인터페이스", r"연결",
            r"배포", r"릴리즈", r"런칭", r"오픈",

            # 수정/개선 동사
            r"수정", r"변경", r"업데이트", r"패치",
            r"추가", r"삭제", r"제거", r"삽입",
            r"리팩토링", r"리팩터링", r"정리", r"클린업",
            r"최적화", r"튜닝", r"개선",

            # 테스트/검증 동사
            r"테스트", r"검증", r"확인", r"검토", r"리뷰",
            r"QA", r"디버깅", r"디버그", r"트러블슈팅",

            # 협업/소통 동사
            r"협업", r"협력", r"소통", r"커뮤니케이션",
            r"회의", r"미팅", r"논의", r"토론", r"제안",
            r"공유", r"전달", r"보고", r"발표",

            # 기술 스택 (프레임워크/언어/도구)
            # Frontend
            r"React", r"Vue", r"Angular", r"Svelte", r"Next", r"Nuxt",
            r"JavaScript", r"TypeScript", r"jQuery", r"HTML", r"CSS", r"SCSS", r"Sass",
            r"Redux", r"Recoil", r"Zustand", r"MobX", r"Vuex", r"Pinia",
            r"Webpack", r"Vite", r"Babel", r"ESLint", r"Prettier",

            # Backend
            r"Spring", r"SpringBoot", r"Spring Boot", r"JPA", r"Hibernate",
            r"Django", r"Flask", r"FastAPI", r"Express", r"NestJS", r"Koa",
            r"Node", r"NodeJS", r"Node\.js", r"Deno",
            r"Ruby", r"Rails", r"Laravel", r"PHP", r"ASP\.NET", r"\.NET",

            # Languages
            r"Java", r"Python", r"Kotlin", r"Swift", r"Go", r"Golang", r"Rust",
            r"C\+\+", r"C#", r"Scala", r"Clojure",

            # Database
            r"MySQL", r"PostgreSQL", r"Postgres", r"MariaDB", r"Oracle", r"MSSQL",
            r"MongoDB", r"Redis", r"Elasticsearch", r"DynamoDB", r"Cassandra",
            r"SQL", r"NoSQL", r"쿼리", r"인덱스", r"정규화",

            # DevOps/Infra
            r"AWS", r"Azure", r"GCP", r"클라우드", r"Cloud",
            r"Docker", r"Kubernetes", r"K8s", r"컨테이너",
            r"Jenkins", r"GitLab", r"GitHub Actions", r"CI/CD", r"CI", r"CD",
            r"Terraform", r"Ansible", r"Nginx", r"Apache",

            # Mobile
            r"Android", r"iOS", r"Flutter", r"React Native", r"Xamarin",

            # etc
            r"Git", r"SVN", r"Jira", r"Confluence", r"Slack", r"Notion",
            r"Figma", r"Zeplin", r"Swagger", r"Postman",
            r"REST", r"GraphQL", r"gRPC", r"WebSocket", r"소켓",
            r"JWT", r"OAuth", r"인증", r"보안",
            r"로그", r"모니터링", r"알림", r"대시보드",
        ]
        has_action = any(re.search(p, answer, re.IGNORECASE) for p in action_patterns)

        # ========================================
        # Result 패턴 (결과/성과/완료/효과)
        # ========================================
        result_patterns = [
            # 완료/성공 표현
            r"완료", r"완성", r"마무리", r"종료", r"끝",
            r"성공", r"성공적", r"무사히", r"정상적",
            r"출시", r"오픈", r"런칭", r"배포", r"릴리즈",
            r"납품", r"인수인계", r"이관",

            # 성과/결과 표현
            r"결과", r"성과", r"효과", r"영향", r"임팩트",
            r"달성", r"도달", r"충족", r"만족",
            r"해결", r"처리", r"극복", r"돌파",

            # 개선/향상 표현
            r"개선", r"향상", r"증가", r"상승", r"올라",
            r"감소", r"줄", r"낮추", r"절감", r"단축",
            r"빨라", r"느려", r"좋아", r"나아",

            # 수치/지표 표현
            r"\d+%", r"\d+퍼센트", r"\d+배", r"\d+倍",
            r"\d+명", r"\d+건", r"\d+개", r"\d+회",
            r"\d+초", r"\d+ms", r"\d+분", r"\d+시간",
            r"TPS", r"RPS", r"QPS", r"DAU", r"MAU", r"PV", r"UV",
            r"트래픽", r"처리량", r"응답시간", r"레이턴시", r"latency",

            # 상태 변화 표현
            r"되었", r"됐", r"됨", r"되어", r"돼",
            r"받았", r"얻었", r"가져", r"확보",
            r"인정", r"평가", r"호평", r"좋은 반응",

            # 수상/인정
            r"수상", r"상", r"1등", r"1위", r"우수", r"최우수",
            r"선정", r"채택", r"합격", r"통과",
        ]
        has_result = any(re.search(p, answer) for p in result_patterns)

        # ========================================
        # Reflection 패턴 (성찰/배움/느낌/다짐)
        # ========================================
        reflection_patterns = [
            # 배움/깨달음 표현
            r"배웠", r"배운", r"배우게", r"습득", r"익히",
            r"느꼈", r"느낀", r"느끼게", r"체감",
            r"깨달", r"알게", r"이해하게", r"파악하게",
            r"경험", r"체험", r"겪으면서", r"통해",

            # 성장/발전 표현
            r"성장", r"발전", r"향상", r"늘", r"나아",
            r"역량", r"능력", r"스킬", r"skill",
            r"자신감", r"확신", r"믿음",

            # 반성/부족 표현
            r"부족", r"미흡", r"아쉬", r"후회", r"반성",
            r"몰랐", r"모르", r"실수", r"잘못",
            r"어려", r"힘들", r"어렵",
            r"한계", r"제한", r"문제점",

            # 다짐/계획 표현
            r"다음에는", r"앞으로", r"향후", r"이후",
            r"노력", r"시도", r"도전", r"계획",
            r"목표", r"방향", r"비전",
            r"더 잘", r"더 나은", r"개선하고",

            # 교훈/인사이트 표현
            r"교훈", r"인사이트", r"insight", r"시사점",
            r"깨닫", r"터득", r"체득",
            r"중요성", r"필요성", r"가치",
            r"덕분에", r"계기로", r"기회",

            # 생각/의견 표현
            r"생각", r"판단", r"의견", r"견해",
            r"같습니다", r"봅니다", r"느낍니다",
            r"것 같", r"듯 합니다", r"듯합니다",
        ]
        has_reflection = any(re.search(p, answer) for p in reflection_patterns)

        # ========================================
        # 기여도 분석 (개인 vs 팀)
        # ========================================
        contribution = "unclear"

        # 명확한 개인 기여 표현
        clear_individual = [
            r"저는", r"제가", r"저의", r"제", r"본인",
            r"I ", r"I'm", r"my ", r"mine",
            r"내가", r"나는", r"나의",
            r"직접", r"스스로", r"혼자",
            r"주도", r"리드", r"이끌",
        ]

        # 팀/그룹 표현
        team_expressions = [
            r"저희", r"우리", r"팀", r"그룹", r"조",
            r"we ", r"our ", r"us ",
            r"같이", r"함께", r"협력", r"협업",
            r"동료", r"팀원", r"멤버",
        ]

        has_individual = any(re.search(p, answer, re.IGNORECASE) for p in clear_individual)
        has_team = any(re.search(p, answer, re.IGNORECASE) for p in team_expressions)

        if has_individual and not has_team:
            contribution = "clear"
        elif has_individual and has_team:
            contribution = "mixed"
        elif has_team and not has_individual:
            contribution = "unclear"
        else:
            # 둘 다 없으면 동사 형태로 추정
            if re.search(r"했습니다|했어요|했고|만들었|개발했|구현했", answer):
                contribution = "mixed"  # 주어 생략된 경우

        # ========================================
        # 구체적 증거 추출 (기술명, 수치, 성과)
        # ========================================
        evidence_clips = []

        # 기술 스택 추출
        tech_patterns = r"(React|Vue|Angular|Svelte|Next\.?js|Nuxt|Spring|SpringBoot|Django|Flask|FastAPI|Express|NestJS|Node\.?js|Java|Python|Kotlin|Swift|Go|Rust|TypeScript|JavaScript|MySQL|PostgreSQL|MongoDB|Redis|AWS|Azure|GCP|Docker|Kubernetes|Git|Jenkins|GraphQL|REST)"
        tech_matches = re.findall(tech_patterns, answer, re.IGNORECASE)
        evidence_clips.extend(list(set(tech_matches)))

        # 수치/지표 추출
        num_patterns = [
            r"(\d+(?:\.\d+)?%)",  # 퍼센트
            r"(\d+(?:\.\d+)?배)",  # 배수
            r"(\d+(?:,\d+)?명)",  # 인원
            r"(\d+(?:,\d+)?건)",  # 건수
            r"(\d+(?:,\d+)?개)",  # 개수
            r"(\d+(?:\.\d+)?초)",  # 시간(초)
            r"(\d+(?:\.\d+)?ms)",  # 밀리초
            r"(\d+(?:,\d+)?원)",  # 금액
            r"(\d+(?:\.\d+)?GB)",  # 용량
            r"(\d+(?:\.\d+)?MB)",  # 용량
        ]
        for pattern in num_patterns:
            matches = re.findall(pattern, answer)
            evidence_clips.extend(matches)

        return {
            "starr": {
                "situation": has_situation,
                "task": has_task,
                "action": has_action,
                "result": has_result,
                "reflection": has_reflection
            },
            "contribution": contribution,
            "evidence_clips": evidence_clips[:10]  # 최대 10개
        }

    def _determine_reaction_type(self, analysis: Dict[str, Any]) -> str:
        """
        답변 품질에 따른 면접관 반응 타입 결정
        - satisfied: STARR 4개 이상, 기여도 명확
        - impressed: STARR 5개 완벽, 증거 풍부
        - good: STARR 3개, 괜찮은 수준
        - neutral: STARR 2개
        - concerned: STARR 1개 이하, 기여도 불명확
        - unsatisfied: 거의 내용 없음
        """
        starr = analysis.get("starr", {})
        filled = sum(1 for v in starr.values() if v)
        contribution = analysis.get("contribution", "unclear")
        evidence = analysis.get("evidence_clips", [])
        quality = analysis.get("answer_quality", "fair")

        # 품질 기반 판단
        if quality == "excellent" or (filled >= 4 and contribution == "clear" and len(evidence) >= 2):
            return "impressed"
        elif quality == "good" or (filled >= 3 and contribution in ["clear", "mixed"]):
            return "satisfied"
        elif filled >= 2:
            return "good" if contribution != "unclear" else "neutral"
        elif filled == 1:
            return "concerned"
        else:
            return "unsatisfied"

    def build_probe(self, analysis: Dict[str, Any], role: str, last_question: str, last_answer: str, difficulty: str = "JUNIOR") -> Dict[str, Any]:
        starr = analysis.get("starr", {})

        # Determine strategy based on missing components
        if not starr.get("action"):
            strategy = "clarify_action"
        elif not starr.get("result"):
            strategy = "clarify_result"
        elif analysis.get("contribution") == "unclear":
            strategy = "clarify_contribution"
        elif not starr.get("reflection"):
            strategy = "reflect"
        else:
            strategy = "paraphrase_and_deepen"

        # 답변 품질에 따른 반응 타입 결정
        reaction_type = self._determine_reaction_type(analysis)

        difficulty_instruction = ""
        if difficulty == "SENIOR":
            difficulty_instruction = "Challenge the candidate on their decisions. Ask 'Why did you choose X over Y?' or about trade-offs."
        else:
            difficulty_instruction = "Encourage them to explain their thought process clearly."

        prompt = f"""
        You are a technical interviewer for a {role} position.
        Difficulty: {difficulty}. {difficulty_instruction}

        Context:
        - Previous Question: "{last_question}"
        - Candidate Answer: "{last_answer}"
        - Analysis Status: {json.dumps(analysis, ensure_ascii=False)}
        - Chosen Strategy: {strategy}

        Task:
        Generate a follow-up question in Korean. The question only, no reaction text needed.
        **CRITICAL: NEVER use the words "STAR", "STARR", "스타", or "STAR 방식" in your question.**

        Output JSON:
        {{
            "next_question": "The follow-up question in Korean",
            "probe_goal": "Goal of this follow-up",
            "requested_evidence": ["list of items to verify"]
        }}
        """

        response_text = self._call_llm(prompt)
        data = self._parse_json_response(response_text)

        # Fallback if LLM fails
        if not data:
            return {
                "next_question": "구체적으로 어떤 기술적 어려움이 있었는지 설명해 주세요.",
                "reaction": {"type": reaction_type, "text": ""},
                "probe_goal": "상세 내용 확인",
                "requested_evidence": []
            }

        return {
            "next_question": data.get("next_question"),
            "reaction": {"type": reaction_type, "text": ""},  # text 비움 - 프론트에서 타입만 사용
            "probe_goal": data.get("probe_goal"),
            "requested_evidence": data.get("requested_evidence", [])
        }

    def _score_from_starr(self, starr: Dict[str, bool]) -> float:
        score = 3.0
        if starr.get("situation"): score += 0.3
        if starr.get("task"): score += 0.3
        if starr.get("action"): score += 0.5
        if starr.get("result"): score += 0.5
        if starr.get("reflection"): score += 0.4
        return min(5.0, score)

    def build_report(self, role: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Using LLM to generate qualitative feedback is better, but keeping it simple for now to align with existing frontend expectations
        # that roughly map to the previous structure.
        
        starr = analysis.get("starr", {})
        score = self._score_from_starr(starr)
        
        prompt = f"""
        Generate a brief feedback comment (Korean) for a candidate based on this analysis:
        Analysis: {json.dumps(analysis, ensure_ascii=False)}
        Score: {score}/5.0
        
        Output JSON:
        {{
            "feedback_level": "High" | "Mid" | "Low",
            "feedback_comment": "One sentence summary"
        }}
        """
        response = self._parse_json_response(self._call_llm(prompt))
        
        return {
            "role": role,
            "competency_scores": { "general": score }, # Simplified for now
            "starr_coverage": starr,
            "individual_contribution": analysis.get("contribution"),
            "strengths": analysis.get("evidence_clips", []), # Using evidence as strengths for now
            "gaps": [k for k, v in starr.items() if not v],
            "feedback_level": response.get("feedback_level", "Mid"),
            "feedback_comment": response.get("feedback_comment", "전반적으로 괜찮으나 구체적인 내용이 더 필요합니다."),
            "evidence_clips": analysis.get("evidence_clips", [])
        }

    def generate_response(self, resume_input: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], last_answer: Optional[str], portfolio_files: Optional[List[str]] = None, total_turns: int = 5, chat_history: Optional[List[Dict[str, Any]]] = None, difficulty: str = "JUNIOR") -> Dict[str, Any]:
        
        # [NEW] Stateless Support: Hydrate State from History
        if chat_history is not None:
            # Deep copy to avoid reference issues
            self.chat_history = [item.copy() for item in chat_history]
            print(f"🔄 Hydrated chat_history from request: {len(self.chat_history)} items")
            
            # [FIX] Defensive Parsing for System messages
            for item in self.chat_history:
                if item.get("role") == "system" and isinstance(item.get("content"), str):
                    try:
                        item["content"] = json.loads(item["content"])
                    except:
                        pass

            # Re-determine Phase based on history
            assistant_questions = [h for h in self.chat_history if h.get("role") == "assistant" and h.get("type") == "question"]
            question_count = len(assistant_questions)
            
            if question_count == 0:
                self.current_phase = "INTRO"
            elif question_count >= total_turns - 1:
                self.current_phase = "CLOSING"
            else:
                self.current_phase = "MAIN"
                
            # [FIX] Restore current_topic_probe_count
            # Scan backwards from the end. Count questions until we hit a "seed" question or start of MAIN.
            # Assuming metadata contains "reaction" type. "transition" or "acknowledge" usually reset the count.
            # If metadata is missing, assumes 0.
            probe_c = 0
            if self.current_phase == "MAIN":
                for item in reversed(self.chat_history):
                    if item.get("role") == "assistant" and item.get("type") == "question":
                        meta = item.get("metadata", {})
                        if isinstance(meta, str): # Handle string metadata if any
                             try: meta = json.loads(meta)
                             except: meta = {}
                        
                        reaction_type = meta.get("reaction", {}).get("type", "")
                        if reaction_type in ["transition", "acknowledge", "welcome"]:
                            # This was a seed question, so we stop counting here (this is count 0 base)
                            break
                        else:
                            # It was a probe/clarify/reflect
                            probe_c += 1
            self.current_topic_probe_count = probe_c
                
            print(f"🔄 State Restored: Phase={self.current_phase}, Question Count={question_count}, Probe Count={self.current_topic_probe_count}")
        
        # 1. Start Interview (INTRO Phase)
        if not self.chat_history:
            self.context["resume"] = resume_input or {}
            self.context["portfolio"] = portfolio or {}
            self.context["total_turns"] = total_turns # ✅ 전체 횟수 저장
            self.context["difficulty"] = difficulty # ✅ 난이도 저장
            
            # --- Portfolio File Parsing ---
            portfolio_text = ""
            files_to_parse = portfolio_files or []
            if not files_to_parse and portfolio and "files" in portfolio:
                files_to_parse = portfolio["files"]
            
            if files_to_parse:
                print(f"📂 Processing {len(files_to_parse)} portfolio files...")
                for file_path in files_to_parse:
                    parsed = FileParser.parse_file(file_path)
                    portfolio_text += f"\n--- File: {os.path.basename(file_path)} ---\n{parsed}\n"
            
            if portfolio_text:
                self.context["portfolio_parsed_text"] = portfolio_text
                print(f"✅ Portfolio Parsed Length: {len(portfolio_text)} chars")
            
            target_role = (
                self.context["resume"].get("classification", {}).get("predicted_role")
                or self.context["resume"].get("target_role")
            )
            self.context["role"] = self.normalize_role(target_role)
            
            # [NEW] Start with INTRO question (자기소개)
            self.current_phase = "INTRO"
            question, probe_goal, requested_evidence = self.build_intro_question(
                self.context["role"], 
                self.context["resume"].get("resume_content"),
                difficulty
            )
            print(f"🎬 [Phase: INTRO] Starting interview with introduction question ({difficulty})")
            
            response_data = {
                "next_question": question,
                "reaction": {
                    "type": "welcome",
                    "text": ""  # 텍스트 비움 - 프론트에서 타입만 사용
                },
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": None,
                "phase": self.current_phase
            }
            
            self.chat_history.append({
                "role": "assistant",
                "type": "question",
                "content": question,
                "metadata": response_data,
                "phase": self.current_phase
            })
            
            return response_data

        # 2. Continue Interview
        
        # [FIX] Prevent Duplicate Answer
        # Check if the last item in history is ALREADY the same as last_answer
        is_duplicate = False
        if self.chat_history and last_answer:
            last_item = self.chat_history[-1]
            if last_item.get("role") == "user" and last_item.get("content") == last_answer:
                is_duplicate = True
                print("⚠️ [State] last_answer already exists in history. Skipping append.")
        
        if last_answer and not is_duplicate:
            self.chat_history.append({
                "role": "user",
                "content": last_answer
            })


        # Get context of previous question
        last_question_item = next((item for item in reversed(self.chat_history) if item["role"] == "assistant"), None)
        last_question_text = last_question_item["content"] if last_question_item else ""

        # Analyze answer
        analysis = self.analyze_answer(last_answer or "")
        self.chat_history.append({
            "role": "system",
            "type": "analysis",
            "content": analysis
        })
        
        # Calculate question count
        assistant_questions = [h for h in self.chat_history if h.get("role") == "assistant" and h.get("type") == "question"]
        question_count = len(assistant_questions)
        
        # Ensure context is loaded if hydrated
        if "role" not in self.context and resume_input:
             target_role = (
                resume_input.get("classification", {}).get("predicted_role")
                or resume_input.get("target_role")
            )
             self.context["role"] = self.normalize_role(target_role)
             self.context["resume"] = resume_input
             self.context["portfolio"] = portfolio
             self.context["difficulty"] = difficulty


        print(f"📊 [Phase: {self.current_phase}] Question #{question_count}, Probe count: {self.current_topic_probe_count}, Difficulty: {difficulty}")

        # [NEW] Phase Transition Logic
        # Case 1: INTRO -> MAIN Transition
        # If we are in INTRO phase limit (question_count=0), current_phase is INTRO.
        # If we hydrated and found 1 question (The intro), current_phase became MAIN.
        # BUT we must treat "Question #1 Answered" as the trigger for the First SEED Question.
        
        if self.current_phase == "INTRO":
             # Legacy path if hydration didn't switch it
             self.current_phase = "MAIN"
             # ...
             
        # [FIX] Implicit Transition check:
        # If we are in MAIN phase, but question_count is exactly 1 (Intro asked),
        # meaning we just finished Intro. We MUST generate the first Seed question.
        # AND we should NOT probe the Intro answer.
        
        is_intro_transition = (question_count == 1)
        
        if self.current_phase == "INTRO" or is_intro_transition:
            # After intro answer, move to MAIN phase
            self.current_phase = "MAIN"
            self.current_topic_probe_count = 0
            print(f"➡️ Transitioning to MAIN phase (Intro Finished)")
            
            # Generate first project question
            # [FIX] Pass previous questions to prevent repeats
            previous_qs = [item["content"] for item in self.chat_history if item.get("role") == "assistant" and item.get("type") == "question"]
            
            question, probe_goal, requested_evidence = self.build_seed_question(
                self.context["role"], 
                self.context["resume"].get("resume_content"), 
                self.context["portfolio"],
                self.context.get("portfolio_parsed_text"),
                difficulty,
                previous_qs # Pass history
            )
            
            # INTRO 답변에 대한 반응 타입 결정
            intro_reaction_type = self._determine_reaction_type(analysis)

            response_data = {
                "next_question": question,
                "reaction": {"type": intro_reaction_type, "text": ""},  # 텍스트 비움
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": self.build_report(self.context.get("role", "backend"), analysis),
                "phase": self.current_phase,
                "analysis_result": analysis  # 분석 결과 포함
            }
            
        elif self.current_phase == "COMPLETED" or question_count >= total_turns - 1:
            self.current_phase = "COMPLETED"
            # Return Goodbye Message
            report = self.build_report(self.context.get("role", "backend"), analysis)
            response_data = {
                "next_question": "면접이 종료되었습니다. 수고하셨습니다. (잠시 후 결과 화면으로 이동합니다)",
                "reaction": {"type": "complete", "text": ""},  # 텍스트 비움
                "probe_goal": "최종 종료",
                "requested_evidence": [],
                "report": report,
                "phase": "COMPLETED",
                "analysis_result": analysis
            }
            self.chat_history.append({
                "role": "assistant",
                "type": "question",
                "content": response_data["next_question"],
                "metadata": response_data,
                "phase": "COMPLETED"
            })
            return response_data

        elif self.current_phase == "CLOSING" or question_count == total_turns - 2:
            self.current_phase = "CLOSING"
            # Return Closing Question
            # 직전 답변에 대한 반응 타입 결정
            closing_reaction_type = self._determine_reaction_type(analysis)

            question, probe_goal, requested_evidence = self.build_closing_question(self.context["role"], difficulty)
            response_data = {
                "next_question": question,
                "reaction": {"type": closing_reaction_type, "text": ""},  # 텍스트 비움
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": self.build_report(self.context.get("role", "backend"), analysis),
                "phase": "CLOSING",
                "analysis_result": analysis
            }
            
        elif self.current_phase == "MAIN":
            # MAIN Phase Logic
            # [NEW] Probe count limiting - prevent infinite follow-ups
            self.current_topic_probe_count += 1
            
            # (Fall through to existing logic below)
                
            starr = analysis.get("starr", {})
            starr_filled = sum(1 for v in starr.values() if v)
            
            # Move to next topic if: probe limit reached OR STARR is sufficiently complete (3+ elements)
            if self.current_topic_probe_count >= self.max_probes_per_topic or starr_filled >= 3:
                print(f"🔄 Moving to next topic (probes: {self.current_topic_probe_count}, STARR filled: {starr_filled})")
                self.current_topic_probe_count = 0
                
                # Generate new topic question
                # [FIX] Pass previous questions
                previous_qs = [item["content"] for item in self.chat_history if item.get("role") == "assistant" and item.get("type") == "question"]

                question, probe_goal, requested_evidence = self.build_seed_question(
                    self.context["role"], 
                    self.context["resume"].get("resume_content"), 
                    self.context["portfolio"],
                    self.context.get("portfolio_parsed_text"),
                    difficulty,
                    previous_qs # Pass history
                )
                
                # 답변 품질에 따른 반응 타입 결정
                transition_reaction_type = self._determine_reaction_type(analysis)

                response_data = {
                    "next_question": question,
                    "reaction": {"type": transition_reaction_type, "text": ""},  # 텍스트 비움
                    "probe_goal": probe_goal,
                    "requested_evidence": requested_evidence,
                    "report": self.build_report(self.context.get("role", "backend"), analysis),
                    "phase": self.current_phase,
                    "analysis_result": analysis  # 분석 결과 포함
                }
            else:
                # Continue probing same topic
                response_data_dict = self.build_probe(
                    analysis, 
                    self.context.get("role", "backend"), 
                    last_question_text, 
                    last_answer,
                    difficulty
                )
                
                response_data = {
                    "next_question": response_data_dict["next_question"],
                    "reaction": response_data_dict["reaction"],
                    "probe_goal": response_data_dict["probe_goal"],
                    "requested_evidence": response_data_dict["requested_evidence"],
                    "report": self.build_report(self.context.get("role", "backend"), analysis),
                    "phase": self.current_phase,
                    "analysis_result": analysis  # 분석 결과 포함
                }

        elif self.current_phase == "CLOSING":
             # Already in closing phase, but backend requested another turn?
             # Just return a polite closing message.
             report = self.build_report(self.context.get("role", "backend"), analysis)
             response_data = {
                "next_question": "면접이 종료되었습니다. 수고하셨습니다.",
                "reaction": {"type": "complete", "text": ""},  # 텍스트 비움
                "probe_goal": "면접 종료",
                "requested_evidence": [],
                "report": report,
                "phase": self.current_phase
            }

        response_data = {
            "next_question": response_data["next_question"],
            "reaction": response_data["reaction"],
            "probe_goal": response_data.get("probe_goal"),
            "requested_evidence": response_data.get("requested_evidence"),
            "report": response_data.get("report"),
            "phase": self.current_phase,
            "analysis_result": analysis # ✅ [NEW] Return raw analysis for stateless storage
        }

        self.chat_history.append({
            "role": "assistant",
            "type": "question",
            "content": response_data["next_question"],
            "metadata": response_data,
            "phase": self.current_phase
        })

        return response_data

    def finalize_interview(self, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # [NEW] Stateless Support
        if chat_history is not None:
             self.chat_history = [item.copy() for item in chat_history]
             print(f"🔄 Hydrated chat_history for Finalize: {len(self.chat_history)} items")

        print(f"🏁 [Finalize] chat_history length: {len(self.chat_history)}")
        print(f"🏁 [Finalize] chat_history roles: {[item.get('role') for item in self.chat_history]}")

        if not self.chat_history:
            print("❌ [Finalize] No chat history!")
            return {"error": "No interview history found."}

        analyses = []
        for item in self.chat_history:
            if item.get("role") == "system" and item.get("type") == "analysis":
                content = item.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        continue
                if isinstance(content, dict):
                    analyses.append(content)

        print(f"🏁 [Finalize] Found {len(analyses)} analysis records")

        if not analyses:
            # 분석 데이터가 없을 때 기본값 반환 (에러 대신)
            print("⚠️ [Finalize] No analysis data found. Returning default values.")
            return {
                "total_score": 2.5,
                "result": "Incomplete",
                "stats": {"question_count": 0, "starr_counts": {}},
                "competency_scores": {"종합": 2.5},
                "strengths": [],  # 강점 없음 (참여 자체는 강점 아님)
                "gaps": ["답변 내용이 분석되지 않았습니다. 질문에 대해 구체적인 경험을 설명해주세요."],
                "final_feedback": "면접 답변이 충분히 분석되지 않았습니다. 다음 면접에서는 질문에 대해 구체적인 경험(상황-과제-행동-결과)을 설명해주시면 더 정확한 평가가 가능합니다."
            }

        total_score = 0.0
        starr_counts = {"situation": 0, "task": 0, "action": 0, "result": 0, "reflection": 0}

        for analysis in analyses:
            starr = analysis.get("starr", {})
            total_score += self._score_from_starr(starr)
            for key in starr_counts:
                if starr.get(key):
                    starr_counts[key] += 1

        count = len(analyses)
        avg_score = round(total_score / count, 2)

        # 논문 기반 상세 역량 점수 계산
        competency_scores = self._calculate_competency_scores_detailed(analyses, starr_counts)

        # 논문 기반 강점 식별
        strengths = self._identify_strengths_from_analyses(analyses)

        # 논문 기반 보완점 식별
        gaps = self._identify_gaps_from_analyses(analyses, starr_counts)

        # LLM으로 최종 피드백 생성 (Conversate 논문 기반)
        final_feedback = self._generate_final_feedback(avg_score, competency_scores, strengths, gaps)

        return {
            "total_score": avg_score,
            "result": "Pass" if avg_score >= 3.0 else "Fail",
            "stats": {
                "question_count": count,
                "starr_counts": starr_counts
            },
            "competency_scores": competency_scores,
            "strengths": strengths,
            "gaps": gaps,
            "final_feedback": final_feedback,
            "history_summary": self.chat_history
        }

    def _identify_strengths_from_analyses(self, analyses: List[Dict]) -> List[str]:
        """
        분석 결과에서 강점 식별 (현실적이고 구체적인 피드백)

        원칙:
        - "면접에 참여했다", "답변을 시도함" 같은 당연한 말 절대 금지
        - 실제 STARR 분석 결과에 기반한 구체적 강점만 제시
        - 강점이 없으면 빈 리스트 반환 (정직함이 중요)
        """
        strengths = []
        evidence_clips = []

        # STARR 요소별 카운트
        situation_count = 0
        task_count = 0
        action_count = 0
        result_count = 0
        reflection_count = 0
        clear_contribution_count = 0

        total = len(analyses) if analyses else 0
        if total == 0:
            return []  # 분석 데이터 없으면 빈 리스트

        for analysis in analyses:
            starr = analysis.get("starr", {})
            contribution = analysis.get("contribution", "unclear")
            clips = analysis.get("evidence_clips", [])

            if starr.get("situation"): situation_count += 1
            if starr.get("task"): task_count += 1
            if starr.get("action"): action_count += 1
            if starr.get("result"): result_count += 1
            if starr.get("reflection"): reflection_count += 1
            if contribution == "clear": clear_contribution_count += 1

            if clips:
                evidence_clips.extend(clips)

        # 비율 계산
        situation_rate = situation_count / total
        task_rate = task_count / total
        action_rate = action_count / total
        result_rate = result_count / total
        reflection_rate = reflection_count / total
        contribution_rate = clear_contribution_count / total

        # 강점 식별 (50% 이상 달성 시에만)
        if action_rate >= 0.5 and result_rate >= 0.5:
            strengths.append("행동(Action)과 결과(Result)를 연결하여 논리적으로 설명함")

        if situation_rate >= 0.5 and task_rate >= 0.5:
            strengths.append("상황과 과제를 명확하게 제시하여 맥락 이해가 쉬움")

        if reflection_rate >= 0.5:
            strengths.append("경험에서 배운 점을 잘 표현하여 성장 가능성을 보여줌")

        if contribution_rate >= 0.5:
            strengths.append("'저는/제가'를 사용해 개인 기여를 명확히 구분함")

        # 증거 기반 강점 (실제 기술/수치가 있을 때만)
        unique_evidence = list(set(evidence_clips))
        if len(unique_evidence) >= 5:
            strengths.append(f"구체적인 기술과 수치를 활용함 ({', '.join(unique_evidence[:3])} 등)")
        elif len(unique_evidence) >= 2:
            strengths.append(f"실제 사용한 기술을 언급함 ({', '.join(unique_evidence[:2])})")

        # 전체적으로 우수한 경우
        all_starr_rate = (situation_rate + task_rate + action_rate + result_rate + reflection_rate) / 5
        if all_starr_rate >= 0.6:
            strengths.append("전반적으로 STARR 구조에 맞춰 체계적으로 답변함")

        # 강점이 없으면 빈 리스트 반환 (절대 "참여했다" 같은 말 금지)
        return strengths[:5]  # 최대 5개

    def _identify_gaps_from_analyses(self, analyses: List[Dict], starr_counts: Dict) -> List[str]:
        """
        분석 결과에서 보완점 식별 (구체적이고 실행 가능한 피드백)

        원칙:
        - 추상적인 조언이 아닌 바로 실행 가능한 개선점
        - "구체적인 답변을 준비하세요" 같은 뻔한 말 금지
        - 실제 STARR 분석 기반으로 부족한 부분만 지적
        """
        gaps = []
        total_questions = len(analyses) if analyses else 0

        if total_questions == 0:
            return ["답변 내용이 분석되지 않았습니다."]

        # 비율 계산
        action_rate = starr_counts.get("action", 0) / total_questions
        result_rate = starr_counts.get("result", 0) / total_questions
        situation_rate = starr_counts.get("situation", 0) / total_questions
        task_rate = starr_counts.get("task", 0) / total_questions
        reflection_rate = starr_counts.get("reflection", 0) / total_questions

        # 기여도 분석
        unclear_count = sum(1 for a in analyses if a.get("contribution") == "unclear")
        unclear_rate = unclear_count / total_questions

        # 증거 분석
        total_evidence = sum(len(a.get("evidence_clips", [])) for a in analyses)

        # 우선순위별 보완점 (가장 부족한 것부터, 30% 미만만)
        # Action이 가장 중요
        if action_rate < 0.3:
            gaps.append("'~했습니다'로 끝나는 구체적인 행동 설명 필요 (예: 'React로 컴포넌트를 구현했습니다')")

        # Result도 중요
        if result_rate < 0.3 and len(gaps) < 2:
            gaps.append("행동의 결과나 성과 추가 필요 (예: '배포 완료', '성능 30% 개선')")

        # Situation/Task
        if situation_rate < 0.3 and task_rate < 0.3 and len(gaps) < 2:
            gaps.append("상황/배경 설명 추가 필요 (예: '캡스톤 프로젝트에서 백엔드를 담당했는데...')")

        # 기여도 문제
        if unclear_rate >= 0.6 and len(gaps) < 3:
            gaps.append("'저는/제가'로 시작하여 본인이 직접 한 일을 명확히 구분해 주세요")

        # 구체성 부족 (증거가 전혀 없을 때만)
        if total_evidence == 0 and len(gaps) < 3:
            gaps.append("사용한 기술명이나 수치(예: 'Spring으로', '2주간')를 추가하면 신뢰도가 높아집니다")

        # 성찰 부족 (다른 보완점이 없을 때만)
        if reflection_rate < 0.2 and len(gaps) < 2:
            gaps.append("'이 경험을 통해 ~를 배웠습니다'로 마무리하면 답변이 더 완성됩니다")

        # ✅ [FIX] 보완점이 없지만 점수가 낮을 때 (불합격 예상) - 가장 부족한 요소 지적
        if len(gaps) == 0:
            # 가장 부족한 요소 찾기
            rates = [
                ("행동(Action)", action_rate, "'~했습니다'로 끝나는 구체적인 행동 설명을 더 추가해 보세요"),
                ("결과(Result)", result_rate, "행동의 결과나 성과(수치, 변화)를 포함하면 더 설득력 있습니다"),
                ("상황(Situation)", situation_rate, "상황/배경 설명을 더 구체적으로 해주시면 좋겠습니다"),
                ("성찰(Reflection)", reflection_rate, "경험에서 배운 점이나 느낀 점을 추가하면 좋습니다"),
            ]
            # 가장 낮은 비율 요소 찾기
            min_rate_item = min(rates, key=lambda x: x[1])
            if min_rate_item[1] < 0.7:  # 70% 미만이면 보완점 추가
                gaps.append(min_rate_item[2])

        return gaps[:3]  # 최대 3개

    def _calculate_competency_scores_detailed(self, analyses: List[Dict], starr_counts: Dict) -> Dict[str, float]:
        """
        상세 역량 점수 계산 (논문 기반)

        역량 구분:
        1. 기술 역량: 내용 정확성, 깊이, 구조
        2. 소프트 스킬: 명확성, 자신감, 일관성
        3. STARR 구조: 각 요소별 달성도
        """
        total = len(analyses) if analyses else 1

        # 1. STARR 기반 점수 (각 요소별)
        starr_scores = {
            "상황_인식": round((starr_counts.get("situation", 0) / total) * 5, 1),
            "과제_명확성": round((starr_counts.get("task", 0) / total) * 5, 1),
            "행동_구체성": round((starr_counts.get("action", 0) / total) * 5, 1),
            "결과_지향성": round((starr_counts.get("result", 0) / total) * 5, 1),
            "성찰_깊이": round((starr_counts.get("reflection", 0) / total) * 5, 1),
        }

        # 2. 기여도 점수
        contribution_score = 0
        for analysis in analyses:
            contrib = analysis.get("contribution", "unclear")
            if contrib == "clear":
                contribution_score += 5
            elif contrib == "mixed":
                contribution_score += 3
            else:
                contribution_score += 1
        contribution_avg = round(contribution_score / total, 1) if total > 0 else 0

        # 3. 종합 점수
        starr_avg = sum(starr_scores.values()) / len(starr_scores) if starr_scores else 0
        overall = round((starr_avg * 0.7 + contribution_avg * 0.3), 1)

        return {
            **starr_scores,
            "기여도_명확성": contribution_avg,
            "종합": overall
        }

    def _generate_final_feedback(self, score: float, competency_scores: Dict, strengths: List[str], gaps: List[str]) -> str:
        """
        LLM을 사용해 최종 피드백 생성 (Conversate 논문 기반)
        - 실제 면접관처럼 자연스러운 피드백
        - 구체적인 개선 방안 제시
        - 빈말이나 형식적 칭찬 금지
        """
        if not self.client:
            # LLM 없을 때 기본 피드백 (더 현실적으로)
            if score >= 4.0:
                return "답변에서 구체적인 경험과 성과를 잘 설명해주셨습니다. 특히 본인이 직접 한 일과 그 결과를 명확히 구분한 점이 좋았습니다. 앞으로도 이런 방식으로 답변하시면 됩니다."
            elif score >= 3.0:
                return "전반적으로 괜찮은 답변이었습니다. 다만 일부 답변에서 '구체적으로 무엇을 했는지'가 더 명확했으면 좋겠습니다. 행동과 결과를 수치나 사례로 표현하면 더 설득력 있는 답변이 됩니다."
            else:
                return "답변 시 상황 설명은 좋았지만, 본인이 실제로 한 행동과 그 결과가 더 구체적이었으면 합니다. 다음 면접에서는 '저는 ~을 해서 ~한 결과를 얻었습니다' 형식으로 답변해 보세요."

        try:
            # 강점이 없으면 빈 문자열 (형식적 칭찬 방지)
            strengths_text = "\n".join([f"- {s}" for s in strengths]) if strengths else "(특별한 강점 없음)"
            gaps_text = "\n".join([f"- {g}" for g in gaps]) if gaps else "(특별한 보완점 없음)"

            prompt = f"""당신은 실제 기업의 기술 면접관입니다. 면접 결과에 대해 솔직하고 실용적인 피드백을 작성해주세요.

## 면접 결과
- 종합 점수: {score}/5.0
- 역량별 점수: {json.dumps(competency_scores, ensure_ascii=False)}

## 분석된 강점
{strengths_text}

## 개선이 필요한 영역
{gaps_text}

## 피드백 작성 원칙
1. 빈말이나 형식적인 칭찬 금지 (예: "면접에 참여해주셔서 감사합니다" 같은 말 금지)
2. 실제로 잘한 부분이 있으면 구체적으로 언급
3. 개선점은 바로 실천할 수 있는 구체적인 팁으로 제시
4. 면접관이 실제로 할 법한 자연스러운 어투 사용
5. 2-3문장으로 핵심만 전달

## 점수별 피드백 방향
- 4.0 이상: 잘한 점 위주로, 작은 개선점 하나
- 3.0-3.9: 잘한 점과 개선점 균형있게
- 3.0 미만: 개선점 위주로, 구체적인 연습 방법 제안

한국어로 피드백을 작성해주세요 (JSON 형식 아님, 자연스러운 문장만):"""

            response = self._call_llm(prompt)
            if response:
                # JSON이나 불필요한 형식 제거
                cleaned = response.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                return cleaned
            return "답변 내용을 바탕으로 개선점을 파악하시고, 다음 면접에서는 구체적인 경험과 수치를 포함해 답변해 보세요."
        except Exception as e:
            print(f"⚠️ Failed to generate final feedback: {e}")
            return "답변 내용을 바탕으로 개선점을 파악하시고, 다음 면접에서는 구체적인 경험과 수치를 포함해 답변해 보세요."
