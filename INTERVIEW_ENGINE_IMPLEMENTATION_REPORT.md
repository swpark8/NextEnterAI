# Interview Engine Implementation Report (STARRR-First)

## 개요
본 문서는 `INTERVIEW_ENGINE_SPEC.md`에 정의된 **STARRR(Situation, Task, Action, Result, Reflection)** 구조를 최우선으로 반영하여 구현된 인터뷰 엔진의 로직과 검증 결과를 기술합니다.

## 주요 구현 사항

### 1. STARRR 정밀 파싱 로직 (`app/services/interview_engine.py`)
기존의 단순 패턴 매칭을 넘어, 각 요소별 핵심 키워드를 스펙에 기반하여 강화했습니다.

- **Situation (상황):** 상황, 이슈, 문제, 장애, 부채, 트래픽, 마감, 요구사항, 배경
- **Task (과제):** 목표, 과제, 책임, 담당, 맡았, 역할, 요청
- **Action (행동):** 구현, 설계, 도입, 적용, 개선, 리팩터, 최적화, 개발, 수정, 분석
- **Result (결과):** %, ms, 초, 배, 증가, 감소, 개선, 절감 (단순 '완료', '해결' 제외 -> 수치/변화 중심)
- **Reflection (성찰):** 다음에는, 다르게, 회고, 배운, 교훈, 아쉬, 깨달, 느꼈

### 2. 기여도(Contribution) 분석 로직
지원자의 답변에서 1인칭 주어와 복수 주어의 빈도를 분석하여 개인 기여도를 판단합니다.

- **Clear:** "제가", "저는" 등 1인칭 주어가 주도적일 때
- **Unclear:** "우리팀이", "함께" 등 복수 주어만 존재할 때
- **Mixed:** 두 가지가 혼재되거나 주어가 불분명할 때

### 3. 프로빙(Probing) 우선순위 로직
인터뷰어(AI)는 답변 분석 결과에 따라 다음 순서로 부족한 요소를 집요하게 질문합니다.

1.  **Missing Action (Clarify):** 구체적인 기술적 실행 내용이 없을 때
    - *질문 예:* "구체적으로 어떤 기술적 조치를 취하셨는지 단계별로 설명해 주세요."
2.  **Missing Result (Clarify):** 정량적 성과가 없을 때
    - *질문 예:* "그 결과가 어떤 지표로 개선되었는지 수치로 설명해 주세요."
3.  **Unclear Contribution (Clarify):** 개인 기여도가 불분명할 때
    - *질문 예:* "팀 성과 중에서 지원자님이 직접 기여한 부분을 구체적으로 알려 주세요."
4.  **Missing Reflection (Reflect):** 성찰/회고가 없을 때
    - *질문 예:* "다시 한다면 어떤 부분을 다르게 하실지, 혹은 이 경험을 통해 배운 점은 무엇인가요?"
5.  **All Good (Paraphrase):** 모든 요소 충족 시
    - *질문 예:* (핵심 요약 및 심화 기술 질문)

## 검증 결과

`tests/test_interview_engine.py`를 통해 다음 시나리오들에 대한 검증을 완료했습니다.

| 테스트 케이스 | 상황 | 기대 결과 | 통과 여부 |
|:---:|:---|:---|:---:|
| `test_analyze_answer_full_starr` | 모든 STARR 요소 포함 | Coverage All True, Contribution Clear | ✅ Pass |
| `test_analyze_answer_missing_action` | Action 키워드 누락 | Action False | ✅ Pass |
| `test_build_probe_missing_action` | Action 누락 상태 | `clarify`, "기술 행동 확인" | ✅ Pass |
| `test_build_probe_missing_result` | Result 누락 상태 | `clarify`, "정량 결과 확인" | ✅ Pass |
| `test_build_probe_unclear_contribution` | 주어 불분명 | `clarify`, "개인 기여 확인" | ✅ Pass |
| `test_build_seed_question_priority` | 포트폴리오 유무에 따른 시드 질문 | Portfolio > Resume > Default 순 생성 | ✅ Pass |

## 실행 방법

서버 실행:
```bash
python app/main.py
```

테스트 실행:
```bash
python -m pytest tests/test_interview_engine.py
```
