from transformers import pipeline
import numpy as np

llm_pipeline = pipeline("text-generation", model="openai-community/gpt2")


def relative_similar_chunks(query, model, chunks, index, embeddings, top_k=3):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context_chunks, llm_pipeline):
    context = "\n".join(context_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    return llm_pipeline(prompt)[0]['generated_text']

