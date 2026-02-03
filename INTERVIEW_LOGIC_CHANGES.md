# AI Interview Logic Updates (2026-02-03)

## Objective

To resolve issues with repeated questions, 500 errors at the end of interviews, and to refine the interview termination flow (Turn 6: Last Question, Turn 7: Goodbye).

## Summary of Changes

### 1. Python AI Server (`interview_engine.py`)

#### A. Prevention of Repeated Questions

- **Logic**: Added `previous_questions` list to `build_seed_question`.
- **Prompt Engineering**: Explicitly instructed the LLM to **AVOID** topics/questions present in the `previous_questions` list.
- **State Restoration**: Enhanced `generate_response` logic to rebuild `previous_qs` from chat history on every turn.

#### B. Interview Flow Adjustment (6 Questions + 1 Goodbye)

- **Turn 1**: Intro Question.
- **Turn 2-5**: Main Competency Questions (Probing).
- **Turn 6 (`question_count == total_turns - 2`)**:
  - **Phase**: `CLOSING`
  - **Action**: Generates the "Last Question" tailored to Junior/Senior difficulty.
- **Turn 7 (`question_count >= total_turns - 1`)**:
  - **Phase**: `COMPLETED`
  - **Action**: Returns a hardcoded "Interview Completed" message ("면접이 종료되었습니다...").
- **Bug Fix**: Added a missing `elif self.current_phase == "CLOSING":` block in `generate_response` to prevent `UnboundLocalError` (500 Error) when the user answers the closing question.

### 2. Java Backend (`InterviewService.java` & `InterviewMessage.java`)

#### A. State Management

- **Role.SYSTEM**: Added `SYSTEM` role to `InterviewMessage` enum.
- **Analysis Storage**: In `submitAnswer`, the AI's `analysis_result` is now saved as a `SYSTEM` message. This allows the Python engine to restore its internal state (probes, topics covered) between turns.

#### B. Termination Logic Update

- **Old Logic**: If `currentTurn >= totalTurns` -> Finalize immediately (User never sees Turn 7 message).
- **New Logic**:
  1. Even if `currentTurn` reaches `totalTurns` (7), allow one last request to AI.
  2. AI returns the "Goodbye" message (Turn 7).
  3. Java saves this message.
  4. **THEN** checks `if (turn >= totalTurns)` to call `finalizeInterview` and calculate scores.
  5. Returns `isCompleted=true` along with the "Goodbye" message.

## Expected User Experience

1.  **Main Interview**: AI asks distinct questions without repetition.
2.  **Turn 6**: User receives "마지막 질문입니다..."
3.  **User Answers Turn 6**:
4.  **Completion**: AI replies "면접이 종료되었습니다. 수고하셨습니다."
5.  **Result**: The "View Results" popup/button becomes available immediately (Status: COMPLETED).

## pending / To-Do

- Verify the flow runs smoothly without "Invalid Status" errors in `JobPostingController` (noted in previous logs).
- Ensure Frontend correctly displays the final "Goodbye" message when `isCompleted` is true.
