# project/core/pdf_reader.py
import PyPDF2

class PDFReader:
    @staticmethod
    def extract_text(pdf_file):
        """
        Extracts text from a PDF file.
        
        :param pdf_file: UploadedFile object from Streamlit
        :return: Extracted text as string
        """
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
