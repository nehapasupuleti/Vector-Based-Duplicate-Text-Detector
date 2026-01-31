from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "The food was delicious and the service was great",
    "The meal tasted amazing and the waiter was very polite",
    "Machine learning is my favorite field",
    "Machine learning is my favorite field",
    "The weather is nice today"
]

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


