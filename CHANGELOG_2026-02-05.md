# AI 기업매칭 파싱 문제 해결 (2026-02-05)

## 개요
Java(Spring Boot) → Python(FastAPI) 간 이력서 데이터 전송 시 JSON 파싱 문제 해결

---

## 수정된 파일

### 1. Java (NextEnterBack)

#### `src/main/java/org/zerock/nextenter/ai/resume/service/ResumeAiRecommendService.java`
- `appendJsonField()`: `Map.toString()` 대신 읽기 좋은 텍스트로 변환
- `convertMapToReadableText()`: 섹션별(학력/경력/프로젝트) 맞춤 텍스트 생성
- `appendIfPresent()`: 다양한 키 이름 지원

#### `src/main/java/org/zerock/nextenter/ai/resume/dto/AiRecommendRequest.java`
- `extractTextList()`: 콤마 구분 문자열 처리 추가
- `extractEducationList()`: 더 많은 키 이름 지원 (school, 학교명, University 등)
- `extractCareerList()`: 더 많은 키 이름 지원 (company, 회사명, 기업명 등)
- `extractProjectList()`: 더 많은 키 이름 지원 (title, 활동명, 경험명 등)
- `extractKeyTasks()`: "Key Tasks:" prefix 필터링 추가
- `cleanTaskString()`: 불필요한 prefix 제거
- `isValidTask()`: 유효한 Task인지 검증
- `calculateExperienceYears()`: period에서 경력 년수 자동 계산
- `extractFirstNonNullValue()`: Fallback용 값 추출

#### `src/main/java/org/zerock/nextenter/ai/resume/ResumeAiService.java`
- 상세 디버그 로깅 추가 (education, professional_experience, skills 건수)

---

### 2. Python (NextEnterAI)

#### `app/services/resume_engine.py`
- **TIER_RULES 엄격 적용**: Fallback 로직 제거
- 해당 티어에 기업이 없으면 다른 티어에서 가져오지 않고 스킵
- 경고 로그 출력: `⚠️ [TIER] 'Top' 티어에 추천 가능한 기업 없음 - 스킵`

---

## 해결된 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| raw_text에 `{key=value}` 형식 출력 | `Map.toString()` 사용 | `convertMapToReadableText()` 구현 |
| key_tasks에 "Key Tasks:" 포함 | 필터링 없음 | `cleanTaskString()` 추가 |
| experience_years가 항상 0.0 | period 파싱 안됨 | `calculateExperienceYears()` 구현 |
| 등급별 추천이 TIER_RULES 안 따름 | Fallback 로직 | Fallback 제거, 엄격 적용 |

---

## 티어별 추천 규칙 (TIER_RULES)

```python
TIER_RULES = {
    "S": ["Top", "Top", "Mid"],    # S등급: Top 2개, Mid 1개
    "A": ["Top", "Mid", "Mid"],    # A등급: Top 1개, Mid 2개
    "B": ["Mid", "Mid", "Low"],    # B등급: Mid 2개, Low 1개
    "C": ["Mid", "Low", "Low"],    # C등급: Mid 1개, Low 2개
    "F": ["Low", "Low", "Low"]     # F등급: Low 3개
}
```

---

## 기업 데이터 현황 (company_50_pool.json)

| 티어 | 기업 수 |
|------|--------|
| Top | 5개 |
| Mid | 9개 |
| Low | 6개 |
| **합계** | **20개** |

---

## 테스트 확인사항

1. Spring 터미널에서 `📊 [AI 전송 데이터 상세]` 로그 확인
2. Python 터미널에서 `key_tasks`에 "Key Tasks:" 없는지 확인
3. Python 터미널에서 `experience_years`가 0.0이 아닌지 확인
4. 등급별 추천이 TIER_RULES대로 나오는지 확인
