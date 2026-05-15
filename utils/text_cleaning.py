# project/utils/text_cleaning.py
import re

class TextCleaner:
    @staticmethod
    def clean(text):
        """
        Cleans extracted text.
        
        :param text: Raw text
        :return: Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        return text.strip()
