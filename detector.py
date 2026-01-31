from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open("data.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

THRESHOLD = 0.75

print("Duplicate pairs:\n")

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = cosine_similarity(
            [embeddings[i]], [embeddings[j]]
        )[0][0]

        if score > THRESHOLD:
            print(texts[i])
            print(texts[j])
            print(f"Similarity: {score:.2f}\n")
