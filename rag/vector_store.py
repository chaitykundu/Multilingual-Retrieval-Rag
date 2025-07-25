#Embedding and vector store 

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def embd_chunks(chunks):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks)
    return embeddings

def store_in_faiss(chunks, embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings