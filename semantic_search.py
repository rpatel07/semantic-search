from sentence_transformers import SentenceTransformer
import numpy as np

documents = [
    "Python is a programming language used for AI development.",
    "Machine learning helps computers learn patterns from data.",
    "Canada is located in North America and has 10 provinces.",
    "Transformers are neural network architectures for NLP tasks.",
    "Docker packages applications in containers.",
    "SQL is used to query and manage relational databases.",
    "Deep learning uses neural networks with many layers.",
    "Git is a version control system used by developers.",
    "RAG stands for Retrieval Augmented Generation.",
    "ChromaDB is a vector database for storing embeddings."
] 
query_list = [
    "Python programming language for AI development",
    "How can computers learn from examples?",
    "What is the best pizza recipe?",
    "SQL",
    "Who is the prime minister of Canada?"
]

model=SentenceTransformer("all-MiniLM-L6-v2")
doc_encodes = model.encode(documents)

def cosine_similarity(a,b):
    a = np.asarray(a)
    b = np.asarray(b)

    denominator = np.linalg.norm(a)*np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a,b)/denominator

def quick_search(query,top_k):
    if(top_k<1):
        raise ValueError("Top_k msut be atleast 1")

    query_encode = model.encode(query)

    scores = np.asarray([cosine_similarity(query_encode,doc_encode) for doc_encode in doc_encodes])

    k = min(top_k,len(documents))
    top_indices = np.argsort(scores)[-k:][::-1]

    return [(documents[i],scores[i]) for i in top_indices]

for query in query_list:
    print(f"\nQuery: {query}")

    results = quick_search(query, top_k=2)

    for document, score in results:
        print(f"Score: {score} | {document}")
