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
        - Ask them to explain using the STAR method (Situation, Task, Action, Result)
        - Matches the {difficulty} level complexity
        - For SENIOR: Focus on architecture decisions, leadership, and strategic impact
        - For JUNIOR: Focus on learning experience, problem-solving approach, and growth
        - DO NOT ask the same question twice.
        
        Example formats:
        - Project: "이력서에 [프로젝트명] 프로젝트가 있는데, 이 프로젝트에서 맡으신 역할과 기술적 도전을 STAR 구조로 설명해 주세요."
        - Career: "[회사명]에서 [직책]으로 근무하시면서 가장 큰 성과를 낸 경험을 STAR 방식으로 말씀해 주시겠어요?"
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
            data.get("question", "대표적인 프로젝트 경험을 STAR 구조로 설명해 주세요."),
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
                "또는 회사에 대해 궁금한 점이 있다면 편하게 질문해 주세요."
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
                "앞으로 어떤 방향으로 성장하고 기여하고 싶으신지, "
                "또는 저희에게 궁금한 점이 있으시면 말씀해 주세요."
            )
            return (
                closing_question,
                "성장 비전 및 회사 적합성 확인",
                ["growth mindset", "company fit", "curiosity"]
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
            
            response_data = {
                "next_question": question,
                "reaction": {"type": "acknowledge", "text": "네, 알겠습니다. 이제 본격적으로 면접을 시작하겠습니다."},
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": self.build_report(self.context.get("role", "backend"), analysis),
                "phase": self.current_phase
            }
            
        elif self.current_phase == "COMPLETED" or question_count >= total_turns - 1:
            self.current_phase = "COMPLETED"
            # Return Goodbye Message
            report = self.build_report(self.context.get("role", "backend"), analysis)
            response_data = {
                "next_question": "면접이 종료되었습니다. 수고하셨습니다. (잠시 후 결과 화면으로 이동합니다)",
                "reaction": {"type": "complete", "text": "모든 면접 과정이 끝났습니다."},
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
            question, probe_goal, requested_evidence = self.build_closing_question(self.context["role"], difficulty)
            response_data = {
                "next_question": question,
                "reaction": {"type": "wrap_up", "text": "이제 마지막 질문을 드리겠습니다."},
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
                
                response_data = {
                    "next_question": question,
                    "reaction": {"type": "transition", "text": "좋습니다. 다른 경험에 대해서도 여쭤볼게요."},
                    "probe_goal": probe_goal,
                    "requested_evidence": requested_evidence,
                    "report": self.build_report(self.context.get("role", "backend"), analysis),
                    "phase": self.current_phase
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
                    "phase": self.current_phase
                }

        elif self.current_phase == "CLOSING":
             # Already in closing phase, but backend requested another turn?
             # Just return a polite closing message.
             report = self.build_report(self.context.get("role", "backend"), analysis)
             response_data = {
                "next_question": "면접이 종료되었습니다. 수고하셨습니다.",
                "reaction": {"type": "complete", "text": "면접이 종료되었습니다."},
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
