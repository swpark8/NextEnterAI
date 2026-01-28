import os
import pypdf
import docx

class FileParser:
    @staticmethod
    def parse_file(file_path: str) -> str:
        """
        Parses text from a given file path (PDF or DOCX).
        Handles local paths.
        """
        if not os.path.exists(file_path):
            return f"[Error] File not found: {file_path}"
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                return FileParser._parse_pdf(file_path)
            elif ext == '.docx':
                return FileParser._parse_docx(file_path)
            else:
                return f"[Error] Unsupported file type: {ext}"
        except Exception as e:
            return f"[Error] Failed to parse file {file_path}: {str(e)}"

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        text = []
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
