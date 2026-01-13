# Text processing logic
import io
from pydantic import BaseModel

# 라이브러리가 없으면 에러가 나지 않도록 예외 처리
try:
    from pypdf import PdfReader
    from docx import Document
except ImportError:
    PdfReader = None
    Document = None

class TextProcessor:
    @staticmethod
    def extract_from_file(file_contents: bytes, filename: str) -> str:
        """
        PDF, DOCX, TXT 파일에서 텍스트만 쏙 뽑아냅니다.
        """
        filename = filename.lower()
        text = ""

        # 라이브러리 설치 확인
        if not PdfReader or not Document:
            return "서버 에러: pypdf 또는 python-docx 라이브러리가 설치되지 않았습니다."

        try:
            # 1. PDF 파일 처리
            if filename.endswith(".pdf"):
                pdf_reader = PdfReader(io.BytesIO(file_contents))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # 2. Word(docx) 파일 처리
            elif filename.endswith(".docx"):
                doc = Document(io.BytesIO(file_contents))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            
            # 3. 텍스트(txt) 파일 처리
            elif filename.endswith(".txt"):
                text = file_contents.decode("utf-8")
            
            else:
                return "지원하지 않는 파일 형식입니다. (pdf, docx, txt만 가능)"

        except Exception as e:
            print(f"텍스트 추출 중 에러 발생: {e}")
            return ""

        return text.strip()