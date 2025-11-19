import os
from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.embeddings_file = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        self.document_map = {}

    def generate_embeddings(self, text: str):
        # if input text is empty or contains only whitespace, raise ValueError
        if not text or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace")

        # use encode method of the model to generate embeddings
        embeddings = self.model.encode(sentences=[text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_strings = []
        for i, doc in enumerate(documents):
            # For each document, add an entry to self.document_map where the key is the id of the document and the value is the document itself.

            self.document_map[doc['id']] = i

            # Create a string representation of each document (movie) and store them all in a list. Each string should have this format: f"{doc['title']}: {doc['description']}"
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_strings.append(doc_string)

        # Use the model's encode method on the list of movie strings. Set the show_progress_bar argument on encode to True so you can see the progress (it takes a while). Store the result as self.embeddings.
        self.embeddings = self.model.encode(
            doc_strings, show_progress_bar=True)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
       # check if the saved embeddings file exists in the cache directory
        if os.path.exists(self.embeddings_file):
            self.documents = documents
            # load the embeddings from the cache directory
            self.embeddings = np.load(self.embeddings_file)
            print(
                f"Loaded embeddings from cache directory: {self.embeddings_file}")
            # Verify the length of the embeddings matches the number of documents
            # if it is, return the cached self.embeddings
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        # otherwise, rebuild and return the result

        self.embeddings = self.build_embeddings(documents)
        np.save(self.embeddings_file, self.embeddings)
        print(f"Saved embeddings to cache directory: {self.embeddings_file}")

        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embeddings(query)

        cosine_similarity_list = []
        for i, doc in enumerate(self.documents):
            doc_embedding = self.embeddings[i]
            cs = cosine_similarity(query_embedding, doc_embedding)
            cosine_similarity_list.append((cs, doc))

        # sort the list of tuples by cosine similarity in descending order
        sorted_cosine_similarity_list = sorted(
            cosine_similarity_list, key=lambda x: x[0], reverse=True)

        # slice the top {limit} results
        top_results = sorted_cosine_similarity_list[:limit]

        # build a list of results containing the score, title and description
        results = []
        for score, doc in top_results:
            results.append({
                'score': score,
                'title': doc['title'],
                'description': doc['description']
            })
        return results


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embeddings = ss.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Dimensions: {embeddings.shape[0]}")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    # stop linting errors
    if ss.documents is None or ss.embeddings is None:
        print("No documents loaded")
        return
    print(f"Number of docs:   {len(ss.documents)}")
    print(
        f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def do_search(query, limit=5):
    ss = SemanticSearch()
    ss.load_or_create_embeddings(load_movies())
    return ss.search(query, limit)
