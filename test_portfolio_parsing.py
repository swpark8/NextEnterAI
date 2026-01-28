import os
from app.services.file_parser import FileParser
from docx import Document

def create_dummy_files():
    # Create DOCX
    doc = Document()
    doc.add_paragraph("Project Beta: Built a React frontend with Next.js.")
    doc.save("test_portfolio.docx")

def test_parsing():
    print("\nTesting DOCX Parsing...")
    docx_text = FileParser.parse_file("test_portfolio.docx")
    print(f"DOCX Result: {docx_text.strip()}")

    # Cleanup
    try:
        os.remove("test_portfolio.docx")
    except:
        pass

if __name__ == "__main__":
    create_dummy_files()
    test_parsing()
