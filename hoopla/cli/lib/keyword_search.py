
import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, BM25_K1


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[str, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies: dict[str, Counter[str]] = {}

    # An __add_document(self, doc_id, text) method.

    def __add_document(self, doc_id: str, text: str) -> None:
        # initialize the term frequency counter for this docid
        self.term_frequencies[doc_id] = Counter()

        tokens = tokenize_text(text)
        # unique token set for index
        for token in set(tokens):
            self.index[token].add(doc_id)

        # raw list of tokens for frequency
        for token in tokens:
            # increment the term frequency counter for this token in this docid
            self.term_frequencies[doc_id][token] += 1

    # A get_documents(self, term) method. It should get the set of document IDs for a given token, and return them as a list, sorted in ascending order.
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self) -> None:
        # iterate over all the movies and add them to both the index and the docmap.
        # When adding the movie data to the index with __add_document(), concatenate the title and the description (i.e., f"{m['title']} {m['description']}") and use that as the input text.
        movies = load_movies()
        for m in movies:
            doc_id = f"{m["id"]}"
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

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

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

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    # Add a new get_tf(self, doc_id: str, term: str) -> int method.
    def get_tf(self, doc_id: str, term: str) -> int:
        # tokenize the term arg
        # Be sure to tokenize the term, but assume that there is only one token.
        # If there's more than one, raise an exception.
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise Exception("get_tf() expects a single token")
        term_token = term_tokens[0]

        # check if the term exists in the term frequencies for that docid
        if doc_id not in self.term_frequencies:
            return 0

            # If the term doesn't exist in that document, return 0.
        if term_token not in self.term_frequencies[doc_id]:
            return 0

            # It should return the times the token appears in the document with the given ID.
        return self.term_frequencies[doc_id][term_token]

    def get_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise Exception("get_idf() expects a single token")
        term_token = term_tokens[0]

        # get the number of documents in the index
        doc_count = len(self.docmap)
        # get the number of documents that contain the term
        term_doc_count = len(self.index[term_token])

        result = math.log((doc_count + 1) / (term_doc_count + 1))

        return result

    def get_tfidf(self, doc_id: str, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise Exception("get_bm25_idf() expects a single token")
        term_token = term_tokens[0]

        # Get total number of documents in the index (N)
        doc_count = len(self.docmap)
        # Get total number of documents that contain the term (df)
        term_doc_count = len(self.index[term_token])

        IDF = math.log((doc_count - term_doc_count + 0.5) /
                       (term_doc_count + 0.5) + 1)

        return IDF

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        sat = (tf * (k1 + 1)) / (tf + k1)
        return sat


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
            results.append(invertedIndex.docmap[str(qr)])
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: str, term: str) -> int:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    return invertedIndex.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    return invertedIndex.get_idf(term)


def tfidf_command(doc_id: str, term: str) -> float:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    return invertedIndex.get_tfidf(doc_id, term)


def bm25idf_command(term: str) -> float:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    return invertedIndex.get_bm25_idf(term)


def bm25_tf_command(doc_id: str, term: str, k1: float = BM25_K1) -> float:
    invertedIndex = InvertedIndex()
    invertedIndex.load()
    return invertedIndex.get_bm25_tf(doc_id, term, k1)


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
