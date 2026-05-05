# fantastic-bassoon
Ethan Bysfield CS32 Final Project

Wiki Link Race

A single player (possibly using a client and server) version of "Wikipedia Races," a game where you attempt to get from one wikipedia page to another by only clicking links in the article, specifically not in disambiguations or "see also," in the least number of steps. I will use the Wikipedia api which specifically has methods to pull random pages, page category, page links, and page text. Likely will display the wikipedia page in a wrapper to allow link clicks and track the number of clicks. For each set of random pages attempt to find the shortest link path and potentially track the shortest path from each of the player's choices to display at the end.

REQUIRED Dependencies
The game window and UI are built using the PyQt6 library so prior to running the following will need to be executed in your terminal:
pip install PyQt6 PyQt6-WebEngine
pip install sentence_transformers
pip install numpy

HOW TO PLAY
The easiest way currently is to download the hub_embeddings.npz,wiki_graph.pkl, and the wiki_race.py file and copy its direct download path. Then in your terminal you can execute the file using python3. 

EXTERNAL SOURCES
The code for the GUI was written by prompting Anthropic's Claude Sonnet 4.6 model. Many of the gameplay elements were also created with the help of prompts to the same generative AI model, especially the currently commented out bidirectional BFS. Natural language processing is done using SentenceTransformers library using their sentence-transformers/all-MiniLM-L6-v2 model
Using Danker precomputed wikipedia link map rather than API to find best links .https://danker.s3.amazonaws.com/index.html
