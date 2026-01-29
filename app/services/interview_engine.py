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

    def build_seed_question(self, role: str, resume_content: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], portfolio_text: Optional[str] = None) -> Tuple[str, str, List[str]]:
        # Use LLM to generate a contextual seed question
        prompt = f"""
        You are a technical interviewer for a {role} position.
        
        Resume Summary:
        {json.dumps(resume_content, ensure_ascii=False, indent=2)}

        Portfolio Summary:
        {json.dumps(portfolio, ensure_ascii=False, indent=2)}
        
        Portfolio Parsed Content (PDF/Docx):
        \"\"\"{portfolio_text or "No attached portfolio files."}\"\"\"

        Task:
        Generate a "STAR" (Situation, Task, Action, Result) behavioral interview question based on the candidate's most significant project or experience.
        If the portfolio content is available, prioritize asking about a specific project found in the portfolio files.
        
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

    def generate_response(self, resume_input: Optional[Dict[str, Any]], portfolio: Optional[Dict[str, Any]], last_answer: Optional[str], portfolio_files: Optional[List[str]] = None) -> Dict[str, Any]:
        # 1. Start Interview
        if not self.chat_history:
            self.context["resume"] = resume_input or {}
            self.context["portfolio"] = portfolio or {}
            
            # --- Portfolio File Parsing (New) ---
            portfolio_text = ""
            files_to_parse = portfolio_files or []
            if not files_to_parse and portfolio and "files" in portfolio:
                files_to_parse = portfolio["files"]
            
            if files_to_parse:
                print(f"📂 Processing {len(files_to_parse)} portfolio files...")
                for file_path in files_to_parse:
                    parsed = FileParser.parse_file(file_path)
                    portfolio_text += f"\n--- File: {os.path.basename(file_path)} ---\n{parsed}\n"
            
            # Store parsed text in context for LLM
            if portfolio_text:
                self.context["portfolio_parsed_text"] = portfolio_text
                print(f"✅ Portfolio Parsed Length: {len(portfolio_text)} chars")
            

            
            target_role = self.context["resume"].get("classification", {}).get("predicted_role")
            self.context["role"] = self.normalize_role(target_role)
            
            question, probe_goal, requested_evidence = self.build_seed_question(
                self.context["role"], 
                self.context["resume"].get("resume_content"), 
                self.context["portfolio"],
                self.context.get("portfolio_parsed_text") # Pass parsed text
            )
            
            response_data = {
                "next_question": question,
                "reaction": {
                    "type": "clarify",
                    "text": "안녕하세요, AI 면접관입니다."
                },
                "probe_goal": probe_goal,
                "requested_evidence": requested_evidence,
                "report": None
            }
            
            self.chat_history.append({
                "role": "assistant",
                "type": "question",
                "content": question,
                "metadata": response_data
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

        # Analyze
        analysis = self.analyze_answer(last_answer or "")
        self.chat_history.append({
            "role": "system",
            "type": "analysis",
            "content": analysis
        })

        # Generate Probe
        probe_data = self.build_probe(analysis, self.context.get("role", "backend"), last_question_text, last_answer or "")
        
        # Create Report for this turn
        report = self.build_report(self.context.get("role", "backend"), analysis)
        
        response_data = {**probe_data, "report": report}

        self.chat_history.append({
            "role": "assistant",
            "type": "question",
            "content": probe_data["next_question"],
            "metadata": response_data
        })

        return response_data

    def finalize_interview(self) -> Dict[str, Any]:
        if not self.chat_history:
            return {"error": "No interview history found."}

        analyses = [item["content"] for item in self.chat_history if item.get("role") == "system" and item.get("type") == "analysis"]
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
