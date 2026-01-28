import sys
import os

# Add parent directory to path to import app
sys.path.append(".") 
print(f"PYTHONPATH: {sys.path}")

# MOCK Dependencies
from unittest.mock import MagicMock
sys.modules["dotenv"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

try:
    from app.services.interview_engine import InterviewEngine
except ImportError as e:
    print(f"Import Error: {e}")
    # Try one level up if run from tests dir
    sys.path.append("..")
    from app.services.interview_engine import InterviewEngine

def run_tests():
    print("Running Verification...")
    engine = InterviewEngine()
    
    # 1. Start logic check
    print("[1/3] Testing Start...")
    resume = {"target_role": "backend", "resume_content": {"skills": ["Python"]}}
    # Mocking genai to avoid api key requirement for test if env is missing
    # or just let it fail gracefully if validation handles it.
    # The code handles missing API key by setting model=None, so it should be fine.
    
    res1 = engine.generate_response(resume, None, None)
    
    if len(engine.chat_history) != 1:
        print(f"FAIL: Start History length {len(engine.chat_history)}")
        return
    if engine.chat_history[0]["role"] != "assistant":
        print("FAIL: Start role mismatch")
        return
    print(f"PASS: Started with question: {res1['next_question'][:30]}...")

    # 2. Answer logic check
    print("[2/3] Testing Answer...")
    answer = "상황은(S) 트래픽이 많았고, 저는(I) 캐싱을 도입해서(A) 해결했습니다(R). 다음에는 더 잘하고 싶습니다(Ref)."
    res2 = engine.generate_response(None, None, answer)
    
    # Expected: 
    # [0] Assistant (Seed)
    # [1] User (Answer)
    # [2] System (Analysis)
    # [3] Assistant (Probe)
    
    if len(engine.chat_history) != 4:
         print(f"FAIL: History length mismatch {len(engine.chat_history)}")
         for i, h in enumerate(engine.chat_history):
             print(f"[{i}] {h.get('role')} {h.get('type')}")
         return
         
    if engine.chat_history[1]["role"] != "user":
        print("FAIL: User answer missing")
        return
        
    print("PASS: Answer processed")

    # 3. Report check
    print("[3/3] Testing Report...")
    report = engine.finalize_interview()
    print(f"Report Result: {report.get('result')}")
    print(f"Report Score: {report.get('total_score')}")
    
    if report.get("result") not in ["Pass", "Fail"]:
        print("FAIL: Invalid Result")
        return
        
    print("PASS: Report")
    print("✅ ALL TESTS PASSED")

if __name__ == "__main__":
    run_tests()
