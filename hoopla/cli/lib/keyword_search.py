
import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path
        self.term_frequencies = dict[int, Counter]

    # An __add_document(self, doc_id, text) method.

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    # A get_documents(self, term) method. It should get the set of document IDs for a given token, and return them as a list, sorted in ascending order.
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self) -> None:
        # iterate over all the movies and add them to both the index and the docmap.
        # When adding the movie data to the index with __add_document(), concatenate the title and the description (i.e., f"{m['title']} {m['description']}") and use that as the input text.
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    # A save method. It should save the index and docmap attributes to disk using the pickle module's dump function.
    def save(self) -> None:
        # Make sure the cache directory exists.
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Use the file path/name cache/index.pkl for the index.
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        # Use the file path/name cache/docmap.pkl for the docmap.
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    # Add a load() method to your InvertedIndex class. It should load the index and docmap from disk using the pickle module's load function.
    def load(self) -> None:
        # Use cache/index.pkl for the index
        # It should raise an error if the files don't exist.
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"{self.index_path} not found")

        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError("f{self.docmap_path} not found")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        # Use cache/docmap.pkl for the docmap
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def build_command() -> None:
    invertedIndex = InvertedIndex()
    invertedIndex.build()
    invertedIndex.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    results = []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        queryResult = invertedIndex.get_documents(qt)
        for qr in queryResult:
            results.append(invertedIndex.docmap[qr])
            if len(results) >= limit:
                return results

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
