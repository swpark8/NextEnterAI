import pytest
import sys
import os

# App 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.interview_engine import InterviewEngine

@pytest.fixture
def engine():
    return InterviewEngine()

def test_analyze_answer_full_starr(engine):
    answer = "당시 트래픽 급증 상황에서(S) 저는(I) DB 인덱싱을 최적화하는 역할을 맡았고(T), 쿼리 실행 계획을 분석하여 커버링 인덱스를 적용했습니다(A). 그 결과 쿼리 속도가 50% 개선되었습니다(R). 다음에는 캐싱 전략도 함께 고려하고 싶습니다(Ref)."
    analysis = engine.analyze_answer(answer)
    
    assert analysis["starr"]["situation"] is True
    assert analysis["starr"]["task"] is True
    assert analysis["starr"]["action"] is True
    assert analysis["starr"]["result"] is True
    assert analysis["starr"]["reflection"] is True
    assert analysis["contribution"] == "clear"

def test_analyze_answer_missing_action(engine):
    answer = "트래픽이 많아서 문제였습니다. 팀원들과 함께 해결했습니다. 결과적으로 빨라졌습니다."
    analysis = engine.analyze_answer(answer)
    
    assert analysis["starr"]["situation"] is True # 문제
    assert analysis["starr"]["action"] is False # 구체적 행동 키워드 없음
    assert analysis["starr"]["result"] is False # 구체적 수치 없음 (빨라졌습니다 -> X)
    assert analysis["contribution"] == "unclear" # "팀원들과 함께", "저" 없음

def test_build_probe_missing_action(engine):
    analysis = {
        "starr": {"situation": True, "task": True, "action": False, "result": True, "reflection": True},
        "contribution": "clear"
    }
    rtype, rtext, goal, evidence = engine.build_probe(analysis)
    assert rtype == "clarify"
    assert goal == "기술 행동 확인"

def test_build_probe_missing_result(engine):
    analysis = {
        "starr": {"situation": True, "task": True, "action": True, "result": False, "reflection": True},
        "contribution": "clear"
    }
    rtype, rtext, goal, evidence = engine.build_probe(analysis)
    assert rtype == "clarify"
    assert goal == "정량 결과 확인"

def test_build_probe_unclear_contribution(engine):
    analysis = {
        "starr": {"situation": True, "task": True, "action": True, "result": True, "reflection": True},
        "contribution": "unclear"
    }
    rtype, rtext, goal, evidence = engine.build_probe(analysis)
    assert rtype == "clarify"
    assert goal == "개인 기여 확인"

def test_build_probe_missing_reflection(engine):
    analysis = {
        "starr": {"situation": True, "task": True, "action": True, "result": True, "reflection": False},
        "contribution": "clear"
    }
    rtype, rtext, goal, evidence = engine.build_probe(analysis)
    assert rtype == "reflect"
    assert goal == "성찰 확인"

def test_build_probe_all_good(engine):
    analysis = {
        "starr": {"situation": True, "task": True, "action": True, "result": True, "reflection": True},
        "contribution": "clear"
    }
    rtype, rtext, goal, evidence = engine.build_probe(analysis)
    assert rtype == "paraphrase"
    assert "심화 탐색" in goal or "이해 확인" in goal

def test_build_seed_question_priority(engine):
    # 1. Portfolio Highlight 우선
    portfolio = {
        "highlights": ["Redis 캐싱 적용"],
        "projects": []
    }
    resume = {"project_experience": [{"project_title": "일반 프로젝트"}]}
    
    q, goal, ev = engine.build_seed_question("backend", resume, portfolio)
    assert "Redis" in q
    assert goal == "포트폴리오 기반 심화 확인"

    # 2. Resume Project 차순위 (Portfolio 없음)
    q2, goal2, ev2 = engine.build_seed_question("backend", resume, None)
    assert "일반 프로젝트" in q2
    assert goal2 == "이력서 프로젝트 기반 질문"

    # 3. Default Seed (둘 다 없음)
    q3, goal3, ev3 = engine.build_seed_question("backend", None, None)
    assert "트래픽" in q3 # Backend default seed keyword

def test_full_interview_flow(engine):
    # 1. Start
    resume = {"target_role": "backend", "resume_content": {"skills": ["Python"]}}
    res1 = engine.generate_response(resume, None, None)
    
    assert len(engine.chat_history) == 1
    assert engine.chat_history[0]["role"] == "assistant"
    assert "next_question" in res1
    
    # 2. User Answer
    answer = "상황은(S) 트래픽이 많았고, 저는(I) 캐싱을 도입해서(A) 해결했습니다(R). 다음에는 더 잘하고 싶습니다(Ref)."
    res2 = engine.generate_response(None, None, answer)
    
    assert len(engine.chat_history) == 4 # Q1, A1, Analysis1, Q2
    assert engine.chat_history[1]["role"] == "user"
    assert engine.chat_history[2]["type"] == "analysis"
    assert engine.chat_history[3]["role"] == "assistant"
    
    # 3. Finalize
    report = engine.finalize_interview()
    assert report["total_score"] > 0
    assert report["result"] in ["Pass", "Fail"]
    assert "feedback" in report
    assert "history_summary" in report
