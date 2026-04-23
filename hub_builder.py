from sentence_transformers import SentenceTransformer, SimilarityFunction
import urllib.request
import urllib.parse
import json
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
WIKI_API = "https://en.wikipedia.org/w/api.php"

HUBS = [
    "United States", "United Kingdom", "World War II", "Europe",
    "Science", "Mathematics", "History", "Geography", "Technology",
    "Politics", "Economics", "Philosophy", "Religion", "Art",
    "Music", "Film", "Literature", "Sport", "Biology", "Physics",
    "Chemistry", "Medicine", "Law", "Language", "Culture"
]

def get_article_text(title):
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1
    })
    req = urllib.request.Request(
        f"{WIKI_API}?{params}",
        headers={"User-Agent": "WikiRaceGame/1.0"}
    )
    data = json.loads(urllib.request.urlopen(req).read())
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", None)
    return None

hub_embeddings = {}
for title in HUBS:
    print(f"Embedding: {title}...")
    text = get_article_text(title)
    if text:
        embedding = model.encode(text)
        hub_embeddings[title] = embedding
titles = list(hub_embeddings.keys())
matrix = np.array(list(hub_embeddings.values()))
np.savez("hub_embeddings.npz", titles=titles, matrix=matrix)