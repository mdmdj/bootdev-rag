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
    print(f"Number of docs:   {len(ss.documents)}")
    print(
        f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")
