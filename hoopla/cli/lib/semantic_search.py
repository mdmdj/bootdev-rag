import os
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json
from .search_utils import CACHE_DIR, load_movies, \
    DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE_LIMIT, \
    CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict] = []
        self.chunk_embeddings_file = CHUNK_EMBEDDINGS_PATH

    def build_chunk_embeddings(self, documents: list[dict]):
        # Populate self.documents and self.document_map from the input documents
        self.documents = documents
        doc_strings = []
        for i, doc in enumerate(documents):
            # For each document, add an entry to self.document_map where the key is the id of the document and the value is the document itself.

            self.document_map[doc['id']] = i

            # Create a string representation of each document (movie) and store them all in a list. Each string should have this format: f"{doc['title']}: {doc['description']}"
            doc_string = f"{doc['title']}: {doc['description']}"
            doc_strings.append(doc_string)

        # create an empty list of strings to hold all the chunks
        self.chunks: list[str] = []

        # create an empty list of dictionaries to hold metadata about each chunk
        self.chunk_metadata: list[dict] = []

        # for each document
        for doc in self.documents:
            # if the description text is empty, skip it
            if not doc['description']:
                continue

            # do sementic chunking on the description text
            chunked_description = semantic_chunk(
                doc['description'], max_chunk_size=4, overlap=1)

            # add each chunk to the all chunks list
            self.chunks.extend(chunked_description)

            # For each chunk, add a dictionary to the "chunk metadata" list with the following keys:

            for i, chunk in enumerate(chunked_description):
                # movie_idx: The index of the document in self.documents
                # chunk_idx: The index of the chunk within the document
                # total_chunks: The total number of chunks in the document
                chunk_metadata = {
                    'movie_idx': self.document_map[doc['id']],
                    'chunk_idx': i,
                    'total_chunks': len(chunked_description)
                }
                self.chunk_metadata.append(chunk_metadata)

        # Use model to encode the entire list of chunks and save the result as self.chunk_embeddings
        self.chunk_embeddings = self.model.encode(
            self.chunks, show_progress_bar=True)

        # save the chunk embeddings to a file
        np.save(self.chunk_embeddings_file, self.chunk_embeddings)

        # save the chunk metadata to a file
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata,
                      "total_chunks": len(self.chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # check if the saved embeddings file exists in the cache directory
        if os.path.exists(self.chunk_embeddings_file):
            self.documents = documents
            # load the embeddings from the cache directory
            self.chunk_embeddings = np.load(self.chunk_embeddings_file)
            print(
                f"Loaded chunk embeddings from cache directory: {self.chunk_embeddings_file}")

            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata_json = json.load(f)

            if len(self.chunk_metadata_json["chunks"]) == self.chunk_metadata_json["total_chunks"]:
                self.chunk_metadata = self.chunk_metadata_json["chunks"]

            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings

        # otherwise, rebuild and return the result
        print("Loaded embeddings don't seem to match\nBuilding chunk embeddings...")
        self.chunk_embeddings = self.build_chunk_embeddings(documents)
        np.save(self.chunk_embeddings_file, self.chunk_embeddings)
        print(
            f"Saved chunk_embeddings to cache directory: {self.chunk_embeddings_file}")

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        # Generate an embedding of the query (using the method from the SemanticSearch class).
        query_embedding = self.generate_embeddings(query)

        # Populate an empty list to store "chunk score" dictionaries
        chunk_scores: list[dict] = []

        # For each chunk embedding:
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            # Calculate the cosine similarity between the chunk embedding and the query embedding.
            cs = cosine_similarity(query_embedding, chunk_embedding)

            # Add a dictionary to the "chunk score" list with the following keys:
            chunk_score = {
                'chunk_idx': self.chunk_metadata[i]['chunk_idx'],
                'movie_idx': self.chunk_metadata[i]['movie_idx'],
                'score': cs
            }

            chunk_scores.append(chunk_score)

        # Create an empty dictionary that maps movie indexes to their scores.
        movie_scores: dict[int, float] = {}

        # For each chunk score, if the movie_idx is not in the movie score dictionary yet, or the new score is higher than the existing one, update the movie score dictionary with the new chunk score.
        for chunk_score in chunk_scores:
            if chunk_score['movie_idx'] not in movie_scores or chunk_score['score'] > movie_scores[chunk_score['movie_idx']]:
                movie_scores[chunk_score['movie_idx']] = chunk_score['score']

        # Sort the movie scores by score in descending order.
        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True)

        # Slice the top {limit} results.
        top_results = sorted_movie_scores[:limit]

        # Build a list of results containing the score, title and description.
        results = []
        for movie_idx, score in top_results:
            results.append({
                # movie id
                'id': self.documents[movie_idx]['id'],
                # movie title
                'title': self.documents[movie_idx]['title'],
                # movie description
                'description': self.documents[movie_idx]['description'],
                # movie score
                'score': score
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


def embed_chunks() -> None:
    css = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


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


def do_search(query, limit=DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    ss.load_or_create_embeddings(load_movies())
    return ss.search(query, limit)


def do_search_chunked(query, limit=DEFAULT_SEARCH_LIMIT):
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(load_movies())
    return css.search_chunks(query, limit)


def chunk(text, chunk_size=DEFAULT_CHUNK_SIZE_LIMIT, overlap=0):
    # split text on white space
    words = text.split()
    # initialize an empty list to store the chunked text
    chunked_text = []

    chunking = True
    start_index = 0
    while chunking:
        # check if we need to rewind the word position by the overlap
        if start_index - overlap > 0:
            start_index -= overlap

        # find the end word position of the chunk by checking the chunk step against the words length
        end_index = min(start_index + chunk_size, len(words))

        chunk = words[start_index:end_index]

        # join the chunk into a single string
        chunk_text = ' '.join(chunk)

        # add the chunk to the chunked_text list
        chunked_text.append(chunk_text)

        # increment the word position by the chunk size
        start_index += chunk_size

        # check if we have reached the end of the text
        if start_index >= len(words):
            chunking = False

    return chunked_text


def semantic_chunk(text, max_chunk_size=4, overlap=0):
    # split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # initialize an empty list to store the chunked text
    chunked_text = []

    chunking = True
    start_index = 0
    while chunking:
        # check if we need to rewind the word position by the overlap
        if start_index - overlap > 0:
            start_index -= overlap

        # find the end word position of the chunk by checking the chunk step against the words length
        end_index = min(start_index + max_chunk_size, len(sentences))

        chunk = sentences[start_index:end_index]

        # join the chunk into a single string
        chunk_text = ' '.join(chunk)

        # add the chunk to the chunked_text list
        chunked_text.append(chunk_text)

        # increment the word position by the chunk size
        start_index += max_chunk_size

        # check if we have reached the end of the text
        if start_index >= len(sentences):
            chunking = False

    return chunked_text
