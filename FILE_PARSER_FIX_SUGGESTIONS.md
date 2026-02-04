# 파일 파서(File Parser) 검토 및 수정 제안 (2026-02-03)

[app/services/file_parser.py](app/services/file_parser.py) – PDF/DOCX 전용 파서 검토 결과. 문제로 보이는 항목만 정리하고 수정 제안을 기록함.

---

## 1. 파일 경로 해석 – 백엔드와 AI 서버 분리 시 File not found

**현재 동작**
- Java 백엔드는 포트폴리오/이력서 저장 시 `FileStorageService.getFileUrl(filename)` 로 **상대 경로**를 반환 (예: `uploads/portfolios/xxx.pdf`).
- 이 경로 문자열이 그대로 AI 요청의 `portfolio_files` / `file_path` 로 Python에 전달됨.
- Python은 `os.path.exists(file_path)` 후 `open(file_path, 'rb')` 로 **로컬 파일 시스템**을 참조함.

**문제**
- AI 서버(Python)와 백엔드(Java)가 **다른 프로세스/컨테이너/서버**이면, Python 쪽 CWD에 `uploads/portfolios/...` 디렉터리가 없음.
- 그 결과 `parse_file` 이 `[Error] File not found: uploads/portfolios/xxx.pdf` 를 반환하고, 포트폴리오/이력서 파일 내용이 AI에 전달되지 않음.

**수정 제안**
- **옵션 A (권장):** 백엔드에서 AI 호출 시 **파일 내용을 base64 등으로 인코딩해 전달**하고, Python은 `parse_file` 대신 “경로 또는 바이너리”를 받는 API로 변경. 경로가 오면 기존처럼 로컬 파일을 열고, 바이너리가 오면 임시 파일 또는 `io.BytesIO` 로 파싱.
- **옵션 B:** Java/Python이 **같은 파일 시스템**을 쓰는 경우에만 유효. 백엔드에서 **절대 경로**를 넘기고, Python에서 `os.path.exists` / `open` 전에 절대 경로인지 확인 후 그대로 사용. (공유 스토리지/마운트 경로가 동일해야 함.)
- **옵션 C:** Python 쪽에 `FILE_PARSER_BASE_DIR` 같은 설정을 두고, 넘어온 경로가 상대 경로일 때 `os.path.join(FILE_PARSER_BASE_DIR, file_path)` 로 해석. 단, 이 디렉터리가 백엔드 업로드 디렉터리와 실제로 동기화되어 있어야 함 (NFS 등).

---

## 2. PDF – 스캔/이미지 PDF에서 텍스트가 비어 있음

**현재 동작**
- `pypdf.PdfReader` + `page.extract_text()` 만 사용. pypdf는 **텍스트 레이어**만 추출함.

**문제**
- **스캔된 PDF**(이미지만 있는 문서)는 텍스트 레이어가 없어 `extract_text()` 가 빈 문자열 또는 거의 비어 있는 문자열을 반환함.
- 예외는 발생하지 않아 `[Error]` 도 나오지 않지만, 결과 문자열은 빈 줄만 많고 **실제 내용이 없음**.

**수정 제안**
- **옵션 A:** 스캔 PDF까지 지원하려면 **OCR** 도입. (예: `pdf2image` + `pytesseract` 로 페이지를 이미지로 띄운 뒤 OCR 텍스트 추출.) 의존성 및 설치 부담이 있음.
- **옵션 B:** OCR 없이 유지하는 경우, `_parse_pdf` 반환값이 **실질적으로 비어 있는지**(예: 공백/줄바꿈만 있는지) 검사해서, 비어 있으면 `"[Warning] No extractable text (e.g. scanned PDF): {file_path}"` 같은 메시지를 반환하도록 하면, 호출부에서 “파싱 실패”로 처리하거나 사용자에게 스캔 PDF는 미지원이라고 안내할 수 있음.

---

## 3. DOCX – 테이블 내용 미추출

**현재 동작**
- `docx.Document(file_path)` 후 **`doc.paragraphs` 만** 사용. `"\n".join([para.text for para in doc.paragraphs])` 로 합침.

**문제**
- **테이블** 안의 텍스트는 `doc.paragraphs` 에 포함되지 않음. 이력서/포트폴리오가 표로 구성된 경우(경력 기간, 스킬 매트릭스 등) **중요 내용이 빠짐**.

**수정 제안**
- `python-docx` 의 `doc.tables` 를 순회해서, 각 셀의 `cell.text` 를 추출해 paragraphs 결과 뒤에 이어 붙이기.
- 예:  
  `tables_text = "\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)`  
  그 다음 `body = "\n".join([p.text for p in doc.paragraphs]); return body + "\n\n" + tables_text` (또는 순서/구분자 정책에 맞게 조정).

---

## 4. 기타 (선택 적용)

- **DOCX 헤더/푸터:** 현재 미추출. 필요하면 `doc.sections` 와 각 section의 `header/footer` 에서 텍스트 추출해 병합 가능.
- **대용량 PDF:** 페이지 수가 매우 많으면 모든 페이지 텍스트를 한 번에 리스트에 넣어 메모리 사용이 커질 수 있음. 필요 시 스트리밍/청크 단위 처리 또는 페이지 수 상한 검토.
- **경로 정규화:** Windows 백슬래시 등이 넘어올 수 있으면 `os.path.normpath(file_path)` 로 정규화 후 존재 여부 확인하면 안전함.
- **의존성:** `requirements.txt` 에 `pypdf`, `python-docx` 명시되어 있음. 버전 고정 시 `pypdf>=x.x`, `python-docx>=x.x` 형태로 두면 호환성 관리에 유리함.

---

## 체크리스트 (반영 시)

- [ ] **1.** 파일 경로: 백엔드–AI 간 파일 전달 방식 정리 (경로 vs 바이너리/base64). 경로만 쓸 경우 절대 경로 또는 Python 기준 BASE_DIR 설정.
- [ ] **2.** PDF: 스캔 PDF 대응 여부 결정. 미지원 시 빈 텍스트일 때 `[Warning]` 반환 처리.
- [ ] **3.** DOCX: `doc.tables` 순회해 테이블 텍스트 추출 후 paragraphs 결과와 병합.
- [ ] **4.** (선택) DOCX 헤더/푸터, 경로 정규화, 대용량 PDF 정책.
