import PyPDF2

reader = PyPDF2.PdfReader(r"C:\Users\Betopia\Desktop\rag_bangla_project\data\HSC26-Bangla1st-Paper.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
print(text[:1000])