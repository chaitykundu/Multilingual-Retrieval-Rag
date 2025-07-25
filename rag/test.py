from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(model.encode(["বাংলা প্রশ্ন"]))
