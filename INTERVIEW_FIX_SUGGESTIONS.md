# AI 면접 엔진 수정 제안 (2026-02-03)

학원 PC에서 반영할 수정 사항 정리. 코드는 수정하지 않고 제안만 기록함.

---

## 1. Python `interview_engine.py` – MAIN 단계에서 `response_data` 미설정 버그

**위치:** [app/services/interview_engine.py](app/services/interview_engine.py) `generate_response` 내 598~634라인 부근

**원인:**  
`elif self.current_phase == "MAIN"` 블록 안에서 `response_data`는 **다음 주제로 넘어가는 경우**에만 설정됨.

- `if self.current_topic_probe_count >= self.max_probes_per_topic or starr_filled >= 3` 일 때만 `response_data` 할당 (627~634).
- 같은 주제로 **probe만 더 하는 경우**(조건 불충족)에는 `response_data`가 설정되지 않음.
- 그 상태로 652라인 `response_data = { "next_question": response_data["next_question"], ... }` 에 도달하면 **UnboundLocalError** → 500 발생.

**수정 제안:**  
MAIN 단계에서 “다음 주제로 넘어가지 않고 probe만 하는” 경로를 추가.

- `current_topic_probe_count < max_probes_per_topic` 이고 `starr_filled < 3` 일 때는 `build_probe(...)` 등으로 **추가 질문**을 생성해 `response_data`를 설정하고, 652라인에 도달하기 전에 그 값을 사용하거나, 652라인에서 `response_data`가 이미 정의된 경우에만 재구성하도록 분기.

---

## 2. `total_turns` 기본값 및 6+1 흐름

**의도:** Turn 1~6 = 질문(Intro + 본문 + 마지막 질문), Turn 7 = “면접이 종료되었습니다. 수고하셨습니다.”

**현재:**
- Python: `generate_response(..., total_turns: int = 5)` (359라인) → 기본값 5.
- Java: `InterviewService` 면접 시작 시 `request.getTotalTurns() != null ? request.getTotalTurns() : 5` (85라인) → 기본값 5.

**수정 제안:**  
6번 질문 + 1번 종료 메시지(총 7턴)를 기본으로 하려면:

- **옵션 A:** Python 기본값을 `total_turns: int = 7` 로 변경.
- **옵션 B:** 면접 시작 API 호출 시 클라이언트/프론트에서 `totalTurns: 7` 을 넘기고, Java는 그대로 사용. (Python에는 Java가 넘기는 `interview.getTotalTurns()` 가 이미 전달되는지 확인.)
- 두 쪽이 맞도록 맞추기: Java에서 `totalTurns` 를 AI 요청에 그대로 넘기고, Python은 인자로 받은 `total_turns` 만 사용하면 됨.

---

## 3. Hydration 시 COMPLETED vs CLOSING

**위치:** 376~382라인

**현재:**  
`question_count >= total_turns - 1` 이면 `current_phase = "CLOSING"` 로 설정.

- `total_turns == 7` 이고 `question_count == 6` 이면 이미 6번째 질문까지 나온 상태이므로, 다음 응답은 “종료 메시지”가 되어야 함.
- 본문 분기에서는 `question_count >= total_turns - 1` 일 때 COMPLETED로 처리하고 있어서, **실제 반환값**은 종료 메시지로 정상 동작.
- 다만 hydration에서 `question_count >= total_turns - 1` 일 때 phase를 `"CLOSING"` 이 아니라 `"COMPLETED"` 로 두는 편이 의미상 더 일치함.

**수정 제안 (선택):**  
Hydration에서 `question_count >= total_turns - 1` 이면 `self.current_phase = "COMPLETED"` 로 설정.

---

## 4. 답변(interview_message) 분석 연계 질문 – 현재 안 됨

**확인 결과:** 사용자 답변은 분석되지만, 그 분석 결과를 사용해 **연계 질문**을 만드는 경로가 호출되지 않음.

**현재 동작:**
- `analyze_answer(last_answer)` 는 매 턴 호출됨 (493라인). STARR·contribution·evidence_clips 등으로 분석함.
- `analysis` 는 `build_report`(피드백), COMPLETED/CLOSING 응답, MAIN 단계의 `starr_filled` 조건 판단에만 사용됨.
- **`build_probe(analysis, role, last_question, last_answer, difficulty)` 는 정의만 되어 있고 `generate_response` 안에서 한 번도 호출되지 않음** (253라인 정의, 호출부 없음).
- MAIN 단계에서 “다음 주제로 넘어가지 않고 같은 주제로 추가 질문(probe)”을 해야 할 때, `build_probe`를 호출하는 `else` 분기가 없어서, 분석 결과를 반영한 연계 질문이 나가지 않음 (그리고 `response_data` 미설정 버그로 500 발생).

**의도된 연계:**  
`build_probe` 는 이전 질문·답변·분석(starr, contribution, strategy)을 받아 “Action/Result/Reflection 보강” 등 전략에 맞는 후속 질문을 LLM으로 생성하도록 되어 있음. 이 경로가 호출되어야 “답변을 분석해서 그에 맞는 다음 질문”이 가능함.

**수정 제안:**  
- MAIN 단계에서 `if self.current_topic_probe_count >= ... or starr_filled >= 3:` 의 **else** 분기 추가.
- 그 안에서 `last_question_text`, `last_answer`, `analysis`, `difficulty` 를 사용해 `build_probe(analysis, self.context["role"], last_question_text, last_answer or "", difficulty)` 를 호출하고, 반환값으로 `response_data` 를 구성 (이때 1번 항목의 `response_data` 미설정 버그도 함께 해소됨).

---

## 5. 참고 – Java 쪽

**저장 순서:**  
`InterviewService.submitAnswer` 에서 “수고하셨습니다” 메시지 저장 → `currentTurn >= totalTurns` 일 때 `finalizeInterview` / `completeInterview` → `isCompleted` 반환 순서는 이미 올바름.  
Python이 500 없이 정상 반환하면, 7번째 메시지 저장 및 완료 처리까지 정상 동작함.

**확인할 것:**  
면접 시작 시 `totalTurns` 를 7로 보내는지(프론트/클라이언트 또는 백엔드 기본값).

---

## 6. 이력서 핵심 카테고리 – 매칭과 동일하게 적용 제안

**배경:** 매칭에서 핵심 평가 카테고리는 **skill, career(professional_experience), 기간, 직무 연관성, education** 임. 면접에서도 같은 축을 쓰면 일관성·공정성·질문 품질이 좋아짐.

**현재 면접 쪽:**  
`_resume_summary_for_prompt` 는 **raw_text, skills, education, professional_experience, project_experience** 를 참고. **기간·직무 연관성**은 “명시된 평가 카테고리”로는 쓰이지 않음 (raw_text/JSON 안에 있으면 간접 반영만 됨).

**수정 제안:**
- **skill, career, education** 는 이미 사용 중이므로 유지.
- **기간:** 이력서(resume_content)에 경력/학력 **기간** 필드가 있으면, `_resume_summary_for_prompt` 또는 질문 생성 프롬프트에서 “N년 차, O년간 P를 했다” 같은 문맥으로 명시적으로 포함. 매칭과 동일한 “기간” 정보를 면접에서도 활용.
- **직무 연관성:** 매칭에서 쓰는 “직무 fit” 개념을 면접에서도 반영. “이 경력/학력/스킬이 지원 직무와 어떻게 연결되는지”를 질문/프롬프트에 명시.
- **project_experience** 는 “경력의 구체적 사례”로 유용하므로 **career 보조**로 유지. 핵심 축은 매칭과 동일하게 skill, career, 기간, 직무 연관성, education 로 통일.

**정리:** 매칭의 핵심 평가 카테고리(skill, career, 기간, 직무 연관성, education)를 면접에서도 **핵심 참고/평가 축**으로 쓰면, 서비스 일관성·공정성·질문 품질 측면에서 유리함.

---



## 체크리스트 (학원 PC에서 적용 시)

- [ ] **1.** `interview_engine.py` MAIN 단계: probe만 할 때도 `response_data` 설정되도록 분기 추가.
- [ ] **2.** MAIN 단계 else 분기에서 **`build_probe(analysis, role, last_question_text, last_answer, difficulty)` 호출**하여 답변 분석 연계 질문이 나가도록 연결.
- [ ] **3.** 6+1 턴 흐름: Python 기본값 또는 클라이언트/Java에서 `totalTurns=7` 사용하도록 통일.
- [ ] **4.** (선택) Hydration에서 `question_count >= total_turns - 1` 이면 `current_phase = "COMPLETED"` 로 설정.
- [ ] **5.** 면접 시작 시 `totalTurns` 가 AI/백엔드에 7로 전달되는지 확인.
- [ ] **6.** 이력서 핵심 카테고리: 기간·직무 연관성 반영 및 매칭과 동일 축(skill, career, education) 통일.
