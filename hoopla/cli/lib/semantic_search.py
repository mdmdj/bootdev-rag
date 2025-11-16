from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(self, text: str):
        # if input text is empty or contains only whitespace, raise ValueError
        if not text or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace")

        # use encode method of the model to generate embeddings
        embeddings = self.model.encode(sentences=[text])
        return embeddings[0]


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
