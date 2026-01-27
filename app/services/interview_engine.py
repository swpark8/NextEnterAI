import re
from typing import Any, Dict, List, Optional, Tuple


class InterviewEngine:
    SEED_QUESTIONS = {
        "backend": [
            {
                "competency": "system_architecture",
                "question": "트래픽 급증으로 시스템 한계에 도달했던 상황을 STAR 구조로 설명해 주세요."
            },
            {
                "competency": "database_consistency",
                "question": "운영 중 데이터 정합성 문제나 데드락을 겪었던 경험이 있다면 어떻게 해결했는지 설명해 주세요."
            },
            {
                "competency": "migration_strategy",
                "question": "레거시 시스템을 새로운 아키텍처로 마이그레이션했던 사례를 STAR로 설명해 주세요."
            },
            {
                "competency": "api_optimization",
                "question": "복잡한 비즈니스 로직 API의 응답 성능을 개선했던 경험을 구체적으로 이야기해 주세요."
            }
        ],
        "frontend": [
            {
                "competency": "performance",
                "question": "Web Vitals 지표가 악화된 문제를 해결했던 경험을 STAR로 설명해 주세요."
            },
            {
                "competency": "state_management",
                "question": "복잡한 전역 상태를 설계하고 최적화했던 경험을 설명해 주세요."
            },
            {
                "competency": "async_handling",
                "question": "사용자 입력에 따른 비동기 요청 경합 문제를 처리했던 사례가 있나요?"
            },
            {
                "competency": "framework_depth",
                "question": "React 훅 사용 중 발생한 까다로운 버그를 디버깅한 경험을 설명해 주세요."
            }
        ],
        "pm": [
            {
                "competency": "prioritization",
                "question": "제한된 리소스에서 우선순위를 결정했던 경험을 STAR로 설명해 주세요."
            },
            {
                "competency": "stakeholder_management",
                "question": "이해관계자 갈등을 조율했던 경험을 구체적으로 이야기해 주세요."
            },
            {
                "competency": "product_sense",
                "question": "실패했던 기능이나 프로젝트에서 배운 점과 개선을 설명해 주세요."
            },
            {
                "competency": "data_analysis",
                "question": "데이터를 근거로 제품 방향을 바꾼 사례가 있나요?"
            }
        ],
        "fullstack": [
            {
                "competency": "end_to_end_design",
                "question": "처음부터 끝까지 설계하고 구현한 기능을 STAR로 설명해 주세요."
            },
            {
                "competency": "api_integration",
                "question": "프론트와 백엔드 간 효율적인 통신을 설계했던 경험을 이야기해 주세요."
            },
            {
                "competency": "debugging",
                "question": "프론트와 백엔드 양쪽에서 동시에 이슈가 발생했을 때의 디버깅 경험을 설명해 주세요."
            },
            {
                "competency": "devops",
                "question": "배포 및 운영 환경에서 장애 대응을 준비했던 사례가 있나요?"
            }
        ],
        "uiux": [
            {
                "competency": "design_process",
                "question": "대표 프로젝트의 디자인 프로세스를 STAR로 설명해 주세요."
            },
            {
                "competency": "data_driven_design",
                "question": "데이터로 디자인 결정을 수정했던 경험을 설명해 주세요."
            },
            {
                "competency": "collaboration",
                "question": "개발 제약으로 디자인 변경이 필요했던 상황을 어떻게 해결했나요?"
            },
            {
                "competency": "design_system",
                "question": "디자인 일관성을 유지하기 위해 어떤 시스템을 구축하거나 활용했나요?"
            }
        ],
        "ai": [
            {
                "competency": "rag",
                "question": "RAG 시스템을 구축했던 경험을 STAR로 설명해 주세요."
            },
            {
                "competency": "fine_tuning",
                "question": "파인튜닝을 수행했던 사례와 그 효과를 설명해 주세요."
            },
            {
                "competency": "prompting_eval",
                "question": "프롬프트 전략과 평가 지표를 설계했던 경험이 있나요?"
            },
            {
                "competency": "safety",
                "question": "유해 응답을 방지하기 위한 가드레일을 어떻게 구현했나요?"
            }
        ]
    }

    ROLE_COMPETENCIES = {
        "backend": ["system_architecture", "database_consistency", "migration_strategy", "api_optimization"],
        "frontend": ["performance", "state_management", "async_handling", "framework_depth"],
        "pm": ["prioritization", "stakeholder_management", "product_sense", "data_analysis"],
        "fullstack": ["end_to_end_design", "api_integration", "debugging", "devops"],
        "uiux": ["design_process", "data_driven_design", "collaboration", "design_system"],
        "ai": ["rag", "fine_tuning", "prompting_eval", "safety"]
    }

    ROLE_KEYWORDS = {
        "backend": ["backend", "server", "api", "db", "database", "spring", "fastapi"],
        "frontend": ["frontend", "react", "ui", "ux", "web", "vite"],
        "pm": ["pm", "product", "manager", "기획"],
        "fullstack": ["fullstack", "full stack"],
        "uiux": ["ui", "ux", "design", "designer", "디자인"],
        "ai": ["ai", "llm", "ml", "nlp", "rag"]
    }

    def normalize_role(self, target_role: Optional[str]) -> str:
        if not target_role:
            return "backend"
        role = target_role.lower()
        for key, keywords in self.ROLE_KEYWORDS.items():
            if any(kw in role for kw in keywords):
                return key
        return "backend"

    def extract_resume_signals(self, resume_content: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        content = resume_content or {}
        skills = []
        skills_data = content.get("skills", {})
        if isinstance(skills_data, dict):
            skills.extend(skills_data.get("essential", []))
            skills.extend(skills_data.get("additional", []))
        projects = content.get("project_experience", []) or []
        tasks = []
        for exp in content.get("professional_experience", []) or []:
            tasks.extend(exp.get("key_tasks", []))
        return {
            "skills": [s for s in skills if isinstance(s, str)],
            "projects": projects,
            "tasks": [t for t in tasks if isinstance(t, str)]
        }

    def extract_portfolio_signals(self, portfolio: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = portfolio or {}
        highlights = data.get("highlights", []) or []
        projects = data.get("projects", []) or []
        return {
            "highlights": [h for h in highlights if isinstance(h, str)],
            "projects": projects
        }

    def _select_portfolio_highlight(self, highlights: List[str], role: str) -> Optional[str]:
        role_keywords = set(self.ROLE_KEYWORDS.get(role, []))
        for highlight in highlights:
            lower = highlight.lower()
            if any(kw in lower for kw in role_keywords):
                return highlight
        return highlights[0] if highlights else None

    def _select_resume_project(self, projects: List[Dict[str, Any]]) -> Optional[str]:
        for project in projects:
            title = project.get("project_title")
            if isinstance(title, str) and title.strip():
                return title.strip()
        return None

    def build_seed_question(self, role: str, resume_content: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
        resume_signals = self.extract_resume_signals(resume_content)
        portfolio_signals = self.extract_portfolio_signals(portfolio)

        highlight = self._select_portfolio_highlight(portfolio_signals["highlights"], role)
        if highlight:
            question = f"포트폴리오에서 '{highlight}'를 언급하셨는데, 이 경험을 STAR 구조로 설명해 주세요."
            return question, "포트폴리오 기반 심화 확인", ["구체적 행동", "정량 결과", "개인 기여"]

        project_title = self._select_resume_project(resume_signals["projects"])
        if project_title:
            question = f"{project_title} 프로젝트에서 본인이 주도한 문제 해결 사례를 STAR로 설명해 주세요."
            return question, "이력서 프로젝트 기반 질문", ["문제 맥락", "기술 선택", "성과 지표"]

        seeds = self.SEED_QUESTIONS.get(role) or self.SEED_QUESTIONS["backend"]
        seed = seeds[0]
        return seed["question"], f"{seed['competency']} 확인", ["구체적 행동", "정량 결과"]

    def _has_pattern(self, text: str, patterns: List[str]) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def analyze_answer(self, answer: str) -> Dict[str, Any]:
        text = answer or ""

        # STARR 패턴 정의 (Spec 기반)
        # Situation: 상황, 이슈, 문제, 장애, 부채, 트래픽, 마감, 요구사항
        starr = {
            "situation": self._has_pattern(text, [r"상황", r"당시", r"이슈", r"문제", r"장애", r"부채", r"트래픽", r"요구사항", r"배경"]),
            "task": self._has_pattern(text, [r"목표", r"과제", r"책임", r"담당", r"맡았", r"역할", r"요청"]),
            "action": self._has_pattern(text, [r"구현", r"설계", r"도입", r"적용", r"개선", r"리팩터", r"최적화", r"개발", r"수정", r"분석"]),
            "result": self._has_pattern(text, [r"%", r"ms", r"초", r"배", r"증가", r"감소", r"개선", r"절감"]),
            "reflection": self._has_pattern(text, [r"다음에는", r"다르게", r"회고", r"배운", r"교훈", r"아쉬", r"깨달", r"느꼈"])
        }

        # 개인 기여도 분석
        # "제가", "저는", "나의" 등 1인칭 주어 카운트
        i_count = len(re.findall(r"\b(저|제|내|나)\b|저는|제가|내가|저의|나의", text))
        # "우리", "팀" 등 복수 주어 카운트
        we_count = len(re.findall(r"우리|팀|함께|동료", text))

        if i_count > 0 and i_count >= we_count:
            contribution = "clear"
        elif i_count == 0 and we_count > 0:
            contribution = "unclear" # "we" only
        elif i_count > 0 and we_count > i_count:
            contribution = "mixed"
        else:
            # 주어가 명확하지 않은 경우, 문맥상 Action 동사가 많으면 clear로 간주할 수도 있으나, 보수적으로 mixed
            contribution = "mixed" 
            if starr["action"] and i_count == 0 and we_count == 0:
                 # 주어 생략된 한국어 특성 고려: Action이 있으면 기여가 있다고 가정하되, 확실하지 않으므로 mixed 유지
                 pass

        evidence_clips = []
        # 숫자 + 단위 패턴 추출 (스펙: evidence_clips)
        for match in re.finditer(r"\d+(?:\.\d+)?\s?(?:%|ms|초|배|억원|만원|건|회|개)?", text):
            raw = match.group(0).strip()
            # 단순 숫자는 제외하고 단위가 있거나 의미있는 숫자만 (간이 로직)
            if raw and raw not in evidence_clips:
                evidence_clips.append(raw)
                
        return {
            "starr": starr,
            "contribution": contribution,
            "evidence_clips": evidence_clips
        }

    def build_probe(self, analysis: Dict[str, Any]) -> Tuple[str, str, str, List[str]]:
        starr = analysis.get("starr", {})
        contribution = analysis.get("contribution")

        # Probing Logic 우선순위 (Spec: Trigger conditions)
        
        # 1. Action이 없으면 -> 기술적 실행 내용 질문
        if not starr.get("action"):
            return "clarify", "구체적으로 어떤 기술적 조치를 취하셨는지 단계별로 설명해 주세요.", "기술 행동 확인", ["핵심 조치", "의사결정 근거", "사용 기술"]
            
        # 2. Result가 없으면 -> 정량적 성과 질문
        if not starr.get("result"):
            return "clarify", "그 결과가 어떤 지표로 개선되었는지 수치로 설명해 주세요.", "정량 결과 확인", ["전후 수치", "영향 범위", "비즈니스 임팩트"]
            
        # 3. 기여도가 불분명하면 -> 개인 기여 질문
        if contribution == "unclear":
            return "clarify", "팀 성과 중에서 지원자님이 직접 기여한 부분을 구체적으로 알려 주세요.", "개인 기여 확인", ["직접 구현", "주도 결정", "역할 분담"]
            
        # 4. Reflection이 없으면 -> 회고 질문
        if not starr.get("reflection"):
            return "reflect", "다시 한다면 어떤 부분을 다르게 하실지, 혹은 이 경험을 통해 배운 점은 무엇인가요?", "성찰 확인", ["개선점", "학습", "아쉬운 점"]
            
        # 5. 모든 요소가 충족되면 -> 정리 및 확인 (Paraphrase)
        return "paraphrase", "말씀하신 내용을 정리하면, 핵심 문제를 주도적으로 해결하여 성과를 냈다는 점이 인상깊습니다. 이 경험에서 가장 큰 기술적 챌린지는 무엇이었나요?", "심화 탐색", ["기술적 난이도", "추가 디테일"]

    def _score_from_starr(self, starr: Dict[str, bool]) -> float:
        score = 3.0
        if starr.get("situation"):
            score += 0.3
        if starr.get("task"):
            score += 0.3
        if starr.get("action"):
            score += 0.5
        if starr.get("result"):
            score += 0.5
        if starr.get("reflection"):
            score += 0.4
        return min(5.0, score)

    def build_report(self, role: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        starr = analysis.get("starr", {})
        coverage_count = sum(1 for v in starr.values() if v)

        if coverage_count >= 4 and analysis.get("contribution") == "clear":
            level = "High"
            comment = "기술적 행동과 결과가 구체적이며 개인 기여가 명확합니다."
        elif coverage_count >= 3:
            level = "Mid"
            comment = "핵심 경험은 전달되나 일부 요소가 부족하여 심화 질문이 필요합니다."
        else:
            level = "Low"
            comment = "경험의 구조화와 정량 근거가 부족합니다."

        strengths = []
        gaps = []
        if starr.get("action"):
            strengths.append("구체적 행동 설명")
        else:
            gaps.append("행동(Action) 디테일 부족")
        if starr.get("result"):
            strengths.append("정량 성과 제시")
        else:
            gaps.append("정량 결과(Result) 부족")
        if not starr.get("reflection"):
            gaps.append("성찰(Reflection) 부족")

        base_score = self._score_from_starr(starr)
        competencies = {c: base_score for c in self.ROLE_COMPETENCIES.get(role, [])}

        return {
            "role": role,
            "competency_scores": competencies,
            "starr_coverage": starr,
            "individual_contribution": analysis.get("contribution"),
            "strengths": strengths,
            "gaps": gaps,
            "feedback_level": level,
            "feedback_comment": comment,
            "evidence_clips": analysis.get("evidence_clips", [])
        }

    def generate_response(self, resume_input: Dict[str, Any], portfolio: Optional[Dict[str, Any]], last_answer: Optional[str]) -> Dict[str, Any]:
        target_role = resume_input.get("classification", {}).get("predicted_role") or resume_input.get("target_role")
        role = self.normalize_role(target_role)

        if not last_answer:
            question, probe_goal, requested_evidence = self.build_seed_question(role, resume_input.get("resume_content"), portfolio)
            return {
                "next_question": question,
                "reaction": {
                    "type": "clarify",
                    "text": "경험을 구체적으로 설명해 주세요."
                },
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": None
            }

        analysis = self.analyze_answer(last_answer)
        reaction_type, reaction_text, probe_goal, requested_evidence = self.build_probe(analysis)
        report = self.build_report(role, analysis)

        return {
            "next_question": reaction_text,
            "reaction": {
                "type": reaction_type,
                "text": reaction_text
            },
            "probe_goal": probe_goal,
            "requested_evidence": requested_evidence,
            "report": report
        }
