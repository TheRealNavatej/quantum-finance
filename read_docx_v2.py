import zipfile
import re

def read_docx(file_path, output_path):
    try:
        with zipfile.ZipFile(file_path) as zf:
            xml_content = zf.read('word/document.xml').decode('utf-8')
            text = re.sub('<[^>]+>', ' ', xml_content)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Successfully wrote to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_docx("quantum qiskit part.docx", "docx_content.txt")
