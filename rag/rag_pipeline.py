import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)