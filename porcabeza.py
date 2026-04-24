import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction
data = np.load("hub_embeddings.npz")
titles = data["titles"].tolist()
matrix = data["matrix"]
hubData = list(zip(titles,matrix))
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
text = "In computer science, iterative deepening search or more specifically iterative deepening depth-first search[1] (IDS or IDDFS) is a state space/graph search strategy in which a depth-limited version of depth-first search is run repeatedly with increasing depth limits until the goal is found. IDDFS is optimal, meaning that it finds the shallowest goal.[2] Since it visits all the nodes in the search tree down to depth the cumulative order in which nodes are first visited is effectively the same as in breadth-first search. However, IDDFS uses much less memory.[1]"
closeHub = ""
closest = 0.0
if text:
    embedding = model.encode(text)
    # Calculates cosine similarity of article with each hub
    for t,m in hubData:
        s = model.similarity(embedding, m).numpy()[0,0]
        if s > closest:
            closest = s
            closeHub = t
print(closest)
print(closeHub)