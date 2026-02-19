import zipfile
import re
import sys

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as zf:
            xml_content = zf.read('word/document.xml').decode('utf-8')
            # very simple regex to strip tags
            text = re.sub('<[^>]+>', ' ', xml_content)
            return text
    except Exception as e:
        return f"Error reading .docx: {e}"

if __name__ == "__main__":
    path = "quantum qiskit part.docx"
    print(read_docx(path))
