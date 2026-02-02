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

    def build_seed_question(self, role: str, resume_content: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], portfolio_text: Optional[str] = None) -> Tuple[str, str, List[str]]:
        # Use LLM to generate a contextual seed question (raw_text fallback when structure is empty)
        resume_summary = self._resume_summary_for_prompt(resume_content)
        prompt = f"""
        You are a technical interviewer for a {role} position.
        
        This is the FIRST question of the interview. There is NO prior conversation.
        
        Resume Summary:
        {resume_summary}

        Portfolio Summary:
        {json.dumps(portfolio, ensure_ascii=False, indent=2)}
        
        Portfolio Parsed Content (PDF/Docx):
        \"\"\"{portfolio_text or "No attached portfolio files."}\"\"\"

        Task:
        Generate an opening interview question in Korean that:
        1. References a SPECIFIC project, experience, or skill from the candidate's resume
        2. Asks them to explain it using the STAR method (Situation, Task, Action, Result)
        3. Does NOT assume any prior conversation or context
        
        Example format: "이력서에 [구체적 프로젝트명/경험]이 있는데, 이 프로젝트에서 어떤 상황(Situation)에서 어떤 과제(Task)를 맡으셨고, 어떻게 해결(Action)하셨는지, 그 결과(Result)는 어땠는지 STAR 구조로 설명해 주시겠어요?"
        
        Output JSON:
        {{
            "question": "The interview question string in Korean",
            "probe_goal": "Short description of what you want to verify (e.g., 'Verification of DB Optimization Experience')",
            "requested_evidence": ["impact metrics", "specific tech stack decision"]
        }}
        """
        
        response_text = self._call_llm(prompt)
        data = self._parse_json_response(response_text)
        
        return (
            data.get("question", "대표적인 프로젝트 경험을 STAR 구조로 설명해 주세요."),
            data.get("probe_goal", "핵심 역량 확인"),
            data.get("requested_evidence", ["구체적 행동", "정량적 성과"])
        )

    def build_intro_question(self, role: str, resume_content: Optional[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
        """Generate an introduction question (self-introduction, motivation)."""
        resume_summary = self._resume_summary_for_prompt(resume_content)
        prompt = f"""
        You are a friendly technical interviewer for a {role} position.
        
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

    def build_closing_question(self, role: str) -> Tuple[str, str, List[str]]:
        """Generate a closing question (future plans, questions for interviewer)."""
        prompt = f"""
        You are a friendly technical interviewer for a {role} position.
        
        The interview is coming to an end. Generate a closing question in Korean.
        
        Task:
        Generate a final question that:
        1. Asks about their future career goals or plans after joining
        2. OR asks if they have any questions for the interviewer
        3. Wraps up the interview in a positive tone
        
        Example: "마지막으로, 입사 후 어떤 개발자로 성장하고 싶으신지, 또는 저희에게 궁금한 점이 있으시면 말씀해 주세요."
        
        Output JSON:
        {{
            "question": "The closing question in Korean",
            "probe_goal": "성장 비전 및 문화 적합성 확인",
            "requested_evidence": ["growth mindset", "curiosity"]
        }}
        """
        
        response_text = self._call_llm(prompt)
        data = self._parse_json_response(response_text)
        
        return (
            data.get("question", "마지막으로, 입사 후 어떤 개발자로 성장하고 싶으신지, 또는 저희에게 궁금한 점이 있으시면 말씀해 주세요."),
            data.get("probe_goal", "성장 비전 확인"),
            data.get("requested_evidence", ["growth mindset", "curiosity"])
        )

    def analyze_answer(self, answer: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the candidate's interview answer based on the STARR method (Situation, Task, Action, Result, Reflection).

        Candidate Answer:
        "{answer}"

        Task:
        1. Identify if each STARR component is present.
        2. Analyze individual contribution (clear "I" vs unclear "We").
        3. Extract specific evidence clips (metrics, technologies).

        Output JSON:
        {{
            "starr": {{
                "situation": true/false,
                "task": true/false,
                "action": true/false,
                "result": true/false,
                "reflection": true/false
            }},
            "contribution": "clear" | "mixed" | "unclear",
            "evidence_clips": ["extracted string 1", "extracted string 2"]
        }}
        """
        
        response_text = self._call_llm(prompt)
        return self._parse_json_response(response_text)

    def build_probe(self, analysis: Dict[str, Any], role: str, last_question: str, last_answer: str) -> Dict[str, Any]:
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

        prompt = f"""
        You are a technical interviewer for a {role} position.
        
        Context:
        - Previous Question: "{last_question}"
        - Candidate Answer: "{last_answer}"
        - Analysis Status: {json.dumps(analysis, ensure_ascii=False)}
        - Chosen Strategy: {strategy}

        Task:
        Generate a follow-up response in Korean.
        
        Output JSON:
        {{
            "next_question": "The follow-up question",
            "reaction": {{
                "type": "clarify" | "reflect" | "paraphrase",
                "text": "A natural conversational reaction (e.g., 'I see, that sounds challenging.')"
            }},
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
                "reaction": {"type": "clarify", "text": "네, 알겠습니다."},
                "probe_goal": "상세 내용 확인",
                "requested_evidence": []
            }
            
        return {
            "next_question": data.get("next_question"),
            "reaction": data.get("reaction"),
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

    def generate_response(self, resume_input: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], last_answer: Optional[str], portfolio_files: Optional[List[str]] = None, total_turns: int = 5) -> Dict[str, Any]:
        # 1. Start Interview (INTRO Phase)
        if not self.chat_history:
            self.context["resume"] = resume_input or {}
            self.context["portfolio"] = portfolio or {}
            self.context["total_turns"] = total_turns # ✅ 전체 횟수 저장
            
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
                self.context["resume"].get("resume_content")
            )
            print(f"🎬 [Phase: INTRO] Starting interview with introduction question")
            
            response_data = {
                "next_question": question,
                "reaction": {
                    "type": "welcome",
                    "text": "안녕하세요, AI 면접관입니다. 편안하게 답변해 주세요."
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
        if last_answer:
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

        # Calculate question count (for phase transitions)
        question_count = len([h for h in self.chat_history if h.get("role") == "assistant" and h.get("type") == "question"])
        total_turns = self.context.get("total_turns", 5)  # Default 5 turns
        
        print(f"📊 [Phase: {self.current_phase}] Question #{question_count}, Probe count: {self.current_topic_probe_count}")

        # [NEW] Phase Transition Logic
        if self.current_phase == "INTRO":
            # After intro answer, move to MAIN phase
            self.current_phase = "MAIN"
            self.current_topic_probe_count = 0
            print(f"➡️ Transitioning to MAIN phase")
            
            # Generate first project question
            question, probe_goal, requested_evidence = self.build_seed_question(
                self.context["role"], 
                self.context["resume"].get("resume_content"), 
                self.context["portfolio"],
                self.context.get("portfolio_parsed_text")
            )
            
            response_data = {
                "next_question": question,
                "reaction": {"type": "acknowledge", "text": "네, 알겠습니다. 이제 본격적으로 면접을 시작하겠습니다."},
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": self.build_report(self.context.get("role", "backend"), analysis),
                "phase": self.current_phase
            }
            
        elif self.current_phase == "MAIN":
            # Check if we should move to CLOSING phase
            if question_count >= total_turns - 1:  # Reserve last turn for closing
                self.current_phase = "CLOSING"
                print(f"➡️ Transitioning to CLOSING phase")
                
                question, probe_goal, requested_evidence = self.build_closing_question(self.context["role"])
                
                response_data = {
                    "next_question": question,
                    "reaction": {"type": "wrap_up", "text": "좋은 답변 감사합니다. 면접이 거의 마무리되어 갑니다."},
                    "probe_goal": probe_goal,
                    "requested_evidence": requested_evidence,
                    "report": self.build_report(self.context.get("role", "backend"), analysis),
                    "phase": self.current_phase
                }
            else:
                # [NEW] Probe count limiting - prevent infinite follow-ups
                self.current_topic_probe_count += 1
                
                starr = analysis.get("starr", {})
                starr_filled = sum(1 for v in starr.values() if v)
                
                # Move to next topic if: probe limit reached OR STARR is sufficiently complete (3+ elements)
                if self.current_topic_probe_count >= self.max_probes_per_topic or starr_filled >= 3:
                    print(f"🔄 Moving to next topic (probes: {self.current_topic_probe_count}, STARR filled: {starr_filled})")
                    self.current_topic_probe_count = 0
                    
                    # Generate new topic question
                    question, probe_goal, requested_evidence = self.build_seed_question(
                        self.context["role"], 
                        self.context["resume"].get("resume_content"), 
                        self.context["portfolio"],
                        self.context.get("portfolio_parsed_text")
                    )
                    
                    response_data = {
                        "next_question": question,
                        "reaction": {"type": "transition", "text": "좋습니다. 다른 경험에 대해서도 여쭤볼게요."},
                        "probe_goal": probe_goal,
                        "requested_evidence": requested_evidence,
                        "report": self.build_report(self.context.get("role", "backend"), analysis),
                        "phase": self.current_phase
                    }
                else:
                    # Continue probing current topic
                    probe_data = self.build_probe(analysis, self.context.get("role", "backend"), last_question_text, last_answer or "")
                    report = self.build_report(self.context.get("role", "backend"), analysis)
                    response_data = {**probe_data, "report": report, "phase": self.current_phase}
                    
        else:  # CLOSING phase
            # Just acknowledge the final answer
            report = self.build_report(self.context.get("role", "backend"), analysis)
            response_data = {
                "next_question": "면접이 종료되었습니다. 오늘 면접에 참여해 주셔서 감사합니다.",
                "reaction": {"type": "complete", "text": "수고하셨습니다!"},
                "probe_goal": "면접 종료",
                "requested_evidence": [],
                "report": report,
                "phase": self.current_phase
            }

        self.chat_history.append({
            "role": "assistant",
            "type": "question",
            "content": response_data["next_question"],
            "metadata": response_data,
            "phase": self.current_phase
        })

        return response_data

    def finalize_interview(self) -> Dict[str, Any]:
        print(f"🏁 [Finalize] chat_history length: {len(self.chat_history)}")
        print(f"🏁 [Finalize] chat_history roles: {[item.get('role') for item in self.chat_history]}")
        
        if not self.chat_history:
            print("❌ [Finalize] No chat history!")
            return {"error": "No interview history found."}

        analyses = [item["content"] for item in self.chat_history if item.get("role") == "system" and item.get("type") == "analysis"]
        print(f"🏁 [Finalize] Found {len(analyses)} analysis records")
        
        if not analyses:
            return {"error": "No analysis data found."}

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
        
        return {
            "total_score": avg_score,
            "result": "Pass" if avg_score >= 3.0 else "Fail",
            "stats": {
                "question_count": count,
                "starr_counts": starr_counts
            },
            "history_summary": self.chat_history
        }
