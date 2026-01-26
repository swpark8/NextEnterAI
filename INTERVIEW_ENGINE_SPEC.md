# Interview Engine Spec

This document defines the interview engine input/output schema and probing logic. The design reuses the existing resume engine scoring model and adds interview-specific preprocessing for STARR coverage and conversational probing.

## Goals
- Reuse resume engine scoring as-is.
- Use portfolio data only for question selection and probing (not for scoring).
- Produce two outputs:
  - Real-time response (next question + reaction + probe intent)
  - Interview report (scores, STARR coverage, feedback)

## Input Schema (Interview Engine)

### Required fields
- `target_role`: string
- `classification.predicted_role`: string (optional fallback)
- `resume_content`: object (same shape as resume_engine input)

### Optional fields
- `evaluation.grade`: string (A/B/C/F etc.)
- `portfolio`: object (used for question selection only)

### Resume content shape (existing)
```
resume_content:
  skills:
    essential: [string]
    additional: [string]
  professional_experience: [
    {
      role: string,
      period: string,
      key_tasks: [string]
    }
  ]
  project_experience: [
    {
      project_title: string,
      description: string
    }
  ]
  education: [
    {
      major: string
    }
  ]
```

### Portfolio shape (new)
```
portfolio:
  links: [string]
  highlights: [string]
  projects: [
    {
      title: string,
      stack: [string],
      impact: string,
      details: string
    }
  ]
```

## Output Schema

### Real-time Response
```
{
  "next_question": string,
  "reaction": {
    "type": "clarify" | "paraphrase" | "reflect",
    "text": string
  },
  "probe_goal": string,
  "requested_evidence": [string]
}
```

### Interview Report
```
{
  "role": string,
  "competency_scores": {
    "performance": number,
    "state_management": number,
    "async_handling": number,
    "framework_depth": number
  },
  "starr_coverage": {
    "situation": boolean,
    "task": boolean,
    "action": boolean,
    "result": boolean,
    "reflection": boolean
  },
  "individual_contribution": "clear" | "mixed" | "unclear",
  "strengths": [string],
  "gaps": [string],
  "feedback_level": "High" | "Mid" | "Low",
  "feedback_comment": string,
  "evidence_clips": [string]
}
```

## Reuse and Mapping Rules
- Scoring is taken from the resume engine output (existing grade logic).
- Interview engine does not change resume scoring rules.
- Portfolio data can only influence:
  - Seed question selection
  - Probing direction
  - Requested evidence

## STARR Parsing Rules (Interview Preprocessing)

Simple heuristics for coverage detection:
- Situation: mentions a specific context or constraint (traffic spike, deadline, incident).
- Task: states ownership or goal ("I was responsible for", "my task was").
- Action: concrete steps or technical decisions (architecture change, indexing, caching, refactor).
- Result: measurable impact (latency, cost, conversion, reliability).
- Reflection: counterfactual or learning ("next time", "I would", "lesson learned").

Individual contribution detection:
- Clear: frequent first-person actions ("I implemented", "I decided")
- Mixed: team actions with partial personal detail
- Unclear: only "we" without personal attribution

## Probing Logic

### Trigger conditions
- Missing Action detail -> Clarify technical steps
- Missing Result detail -> Ask for measurable impact
- Team-only answer -> Clarify personal contribution
- No Reflection -> Ask "what would you do differently"

### Reaction templates
- Clarify: request specific technical detail or bottleneck
- Paraphrase: summarize to confirm understanding
- Reflect: acknowledge pressure and ask about decision criteria

## Seed Question Selection (Portfolio-Aware)

Priority order:
1. Role-specific core competency seed
2. Resume project matching a competency gap
3. Portfolio highlight to deepen technical detail

Portfolio influence rules:
- If portfolio mentions a technique (virtualization, RAG, microservices), use it to generate a probing question.
- If portfolio lacks evidence, use portfolio to ask for proof or metrics.
- Do not use portfolio to increase score directly.

## Example Input
```json
{
  "target_role": "Backend Developer",
  "classification": {
    "predicted_role": "Backend Developer"
  },
  "evaluation": {
    "grade": "A"
  },
  "resume_content": {
    "skills": {
      "essential": ["Python", "Django", "FastAPI", "AWS"],
      "additional": ["Docker", "Git", "Redis"]
    },
    "professional_experience": [
      {
        "role": "Backend Server Developer",
        "period": "2020.01 - 2023.01 (3년)",
        "key_tasks": ["REST API 설계 및 구현", "DB 최적화", "AWS 인프라 관리"]
      }
    ],
    "project_experience": [
      {
        "project_title": "주문 처리 서비스",
        "description": "모놀리식 구조를 마이크로서비스로 전환"
      }
    ],
    "education": [
      {"major": "컴퓨터공학과"}
    ]
  },
  "portfolio": {
    "links": ["https://example.com"],
    "highlights": ["Redis 캐싱으로 응답 지연 개선"],
    "projects": [
      {
        "title": "주문 처리 서비스",
        "stack": ["FastAPI", "Redis"],
        "impact": "p95 450ms -> 210ms",
        "details": "캐시 스탬피드 방지 전략 적용"
      }
    ]
  }
}
```

## Example Real-time Response
```json
{
  "next_question": "모놀리식 구조를 마이크로서비스로 전환했던 상황을 STAR 구조로 설명해 주세요.",
  "reaction": {
    "type": "clarify",
    "text": "성능 개선에 Redis 캐싱이 도움이 되었다고 하셨는데, 병목 구간과 캐시 전략을 구체적으로 설명해 주실 수 있나요?"
  },
  "probe_goal": "기술 디테일과 개인 기여 확인",
  "requested_evidence": ["병목 구간", "캐시 전략", "정량 지표"]
}
```

## Example Interview Report
```json
{
  "role": "Backend Developer",
  "competency_scores": {
    "system_architecture": 4.2,
    "database_consistency": 3.6,
    "migration_strategy": 4.0,
    "api_optimization": 3.8
  },
  "starr_coverage": {
    "situation": true,
    "task": true,
    "action": true,
    "result": true,
    "reflection": false
  },
  "individual_contribution": "clear",
  "strengths": ["마이크로서비스 전환 경험", "정량 성능 지표 제시"],
  "gaps": ["성찰 질문에 대한 답변 부족"],
  "feedback_level": "Mid",
  "feedback_comment": "시스템 전환과 성능 개선 경험이 명확하나, 회고 관점의 설명이 부족합니다.",
  "evidence_clips": ["p95 450ms -> 210ms", "Redis 캐싱 적용"]
}
```

## Notes
- This spec is aligned with the existing resume engine input structure used in `app/services/resume_engine.py`.
- Portfolio data is strictly excluded from scoring.
