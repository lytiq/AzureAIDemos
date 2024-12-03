from promptflow import tool
import numpy as np 
import json

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@tool
def lookup(query_embedding: list, k : int) -> list:
    with open("vectordb.json", 'r') as f:
        vectordb = json.load(f)
        chunks = vectordb['chunks']
        embeddings = vectordb['embeddings']
        
    # Calculate similarity between the user question & each chunk
    similarities = [cosine_similarity(query_embedding, chunk) for chunk in embeddings]
    # print("similarity scores: ", similarities)

    # Get indices of the top k most similar chunks
    sorted_indices = np.argsort(similarities)[::-1]
    scores = np.array(similarities)[sorted_indices]

    # Keep only the top k indices
    top_indices = sorted_indices[:k]
    # print("Here are the indices of the top k chunks after retrieval: ", top_indices)

    # Retrieve the top k most similar chunks
    top_chunks_after_retrieval = [{'page_content' : chunks[i], 'score' : scores[i]} for i in top_indices]
        
    return top_chunks_after_retrieval